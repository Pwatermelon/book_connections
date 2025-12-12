"""
Flask веб-приложение для анализа связей в книгах.
"""
from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import chardet
from werkzeug.utils import secure_filename
from entity_extractor import EntityExtractor
from relation_extractor import RelationExtractor
from ontology_builder import OntologyBuilder
from graph_visualizer import GraphVisualizer
from pathlib import Path
import tempfile


def read_text_file(filepath: str) -> str:
    """
    Читает текстовый файл с автоматическим определением кодировки.
    Пробует несколько кодировок, если не удается определить автоматически.
    
    Args:
        filepath: Путь к файлу
        
    Returns:
        Текст файла в виде строки
    """
    # Список кодировок для попытки чтения
    encodings = ['utf-8', 'windows-1251', 'cp866', 'iso-8859-5', 'utf-8-sig']
    
    # Пробуем определить кодировку автоматически
    try:
        with open(filepath, 'rb') as f:
            raw_data = f.read()
            detected = chardet.detect(raw_data)
            if detected and detected['encoding']:
                encodings.insert(0, detected['encoding'])
    except Exception:
        pass  # Если не удалось определить, используем список по умолчанию
    
    # Пробуем прочитать файл с разными кодировками
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                text = f.read()
                return text
        except (UnicodeDecodeError, LookupError):
            continue
    
    # Если ничего не сработало, пробуем с ошибками
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Не удалось прочитать файл. Возможна проблема с кодировкой: {e}")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB максимум
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Создаем папки, если их нет
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'txt'}

def allowed_file(filename):
    """Проверяет, разрешено ли расширение файла."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Главная страница с формой загрузки."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Обработка загрузки и анализ файла."""
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не загружен'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Разрешены только .txt файлы'}), 400
    
    try:
        # Сохраняем файл
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Читаем текст с определением кодировки
        text = read_text_file(filepath)
        
        if len(text.strip()) == 0:
            return jsonify({'error': 'Файл пуст'}), 400
        
        # Обрабатываем текст
        result = process_book(text, filename)
        
        # Сохраняем результаты (без ontology_builder, т.к. это объект)
        result_id = filename.rsplit('.', 1)[0] + '_' + str(hash(text) % 1000000)
        result_file = os.path.join(app.config['RESULTS_FOLDER'], f'{result_id}.json')
        
        # Создаем копию результата без объекта ontology_builder для JSON
        result_for_json = {k: v for k, v in result.items() if k != 'ontology_builder'}
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_for_json, f, ensure_ascii=False, indent=2)
        
        # Сохраняем граф
        graph_file = os.path.join(app.config['RESULTS_FOLDER'], f'{result_id}_graph.html')
        result['graph_file'] = graph_file
        
        # Сохраняем OWL онтологию
        owl_file = os.path.join(app.config['RESULTS_FOLDER'], f'{result_id}_ontology.owl')
        result['ontology_builder'].export_to_owl(owl_file, ontology_name=filename.rsplit('.', 1)[0])
        result['owl_file'] = owl_file
        
        # Возвращаем результаты
        return jsonify({
            'success': True,
            'result_id': result_id,
            'statistics': result['statistics'],
            'graph_file': f'/results/{result_id}_graph.html',
            'owl_file': f'/results/{result_id}_ontology.owl'
        })
    
    except Exception as e:
        return jsonify({'error': f'Ошибка обработки: {str(e)}'}), 500


def process_book(text: str, filename: str) -> dict:
    """
    Обрабатывает текст книги и возвращает результаты.
    
    Args:
        text: Текст книги
        filename: Имя файла
        
    Returns:
        Словарь с результатами анализа
    """
    # Шаг 1: Извлечение сущностей
    entity_extractor = EntityExtractor()
    entities = entity_extractor.extract_entities(text)
    
    # Шаг 2: Извлечение связей
    relation_extractor = RelationExtractor()
    relations = relation_extractor.extract_relations(text, entities)
    
    # Шаг 3: Построение онтологии
    ontology_builder = OntologyBuilder()
    ontology = ontology_builder.build_ontology(entities, relations)
    statistics = ontology_builder.get_statistics()
    
    # Шаг 4: Визуализация графа
    result_id = filename.rsplit('.', 1)[0] + '_' + str(hash(text) % 1000000)
    graph_file = os.path.join(app.config['RESULTS_FOLDER'], f'{result_id}_graph.html')
    
    graph_visualizer = GraphVisualizer()
    graph = graph_visualizer.build_graph(ontology)
    graph_visualizer.visualize_interactive(output_file=graph_file)
    
    # Подготавливаем данные для отображения
    entities_list = []
    for name, data in ontology['entities'].items():
        entities_list.append({
            'name': name,
            'type': data['type'],
            'mentions': data['attributes'].get('mentions', 0),
            'relations_count': data['attributes'].get('total_relations', 0)
        })
    
    relations_list = []
    for rel in ontology['relations']:
        relations_list.append({
            'source': rel['source'],
            'target': rel['target'],
            'type': rel['type'],
            'confidence': rel.get('confidence', 0.5),
            'context': rel.get('context', '')[:100]
        })
    
    # Преобразуем онтологию для JSON сериализации (set -> list)
    ontology_serializable = {
        'entities': ontology['entities'],
        'relations': ontology['relations'],
        'relation_types': list(ontology['relation_types'])  # Преобразуем set в list
    }
    
    return {
        'filename': filename,
        'text_length': len(text),
        'statistics': statistics,
        'entities': entities_list,
        'relations': relations_list,
        'ontology': ontology_serializable,
        'graph_file': graph_file,
        'ontology_builder': ontology_builder  # Сохраняем для экспорта в OWL (не сериализуется в JSON)
    }


@app.route('/results/<filename>')
def serve_result(filename):
    """Отдает сохраненный граф."""
    filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath)
    return jsonify({'error': 'Файл не найден'}), 404


@app.route('/analyze', methods=['POST'])
def analyze_direct():
    """Прямой анализ текста (без загрузки файла)."""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Текст не предоставлен'}), 400
    
    text = data['text']
    
    if len(text.strip()) == 0:
        return jsonify({'error': 'Текст пуст'}), 400
    
    try:
        result = process_book(text, 'direct_input.txt')
        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        return jsonify({'error': f'Ошибка обработки: {str(e)}'}), 500


if __name__ == '__main__':
    print("=" * 80)
    print("🚀 Запуск веб-приложения для анализа связей в книгах")
    print("=" * 80)
    print(f"📁 Загрузки: {app.config['UPLOAD_FOLDER']}")
    print(f"📊 Результаты: {app.config['RESULTS_FOLDER']}")
    print()
    print("Откройте в браузере: http://127.0.0.1:5000")
    print("=" * 80)
    app.run(debug=True, host='0.0.0.0', port=5000)

