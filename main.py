"""
Основной модуль для запуска программы анализа связей в книге.
"""
import sys
import os
from pathlib import Path
from entity_extractor import EntityExtractor
from relation_extractor import RelationExtractor
from ontology_builder import OntologyBuilder
from graph_visualizer import GraphVisualizer


def load_text_from_file(file_path: str) -> str:
    """
    Загружает текст из файла с автоматическим определением кодировки.
    
    Args:
        file_path: Путь к файлу с текстом
        
    Returns:
        Текст книги
    """
    try:
        # Список кодировок для попытки чтения
        encodings = ['utf-8', 'windows-1251', 'cp866', 'iso-8859-5', 'utf-8-sig']
        
        # Пробуем определить кодировку автоматически
        try:
            import chardet
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Читаем первые 10KB для определения
                detected = chardet.detect(raw_data)
                if detected and detected['encoding']:
                    encodings.insert(0, detected['encoding'])
        except ImportError:
            pass  # chardet не установлен, используем список по умолчанию
        except Exception:
            pass
        
        # Пробуем прочитать файл с разными кодировками
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                    return text
            except (UnicodeDecodeError, LookupError):
                continue
        
        # Если ничего не сработало, пробуем с обработкой ошибок
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Не удалось прочитать файл. Возможна проблема с кодировкой: {e}")
            
    except FileNotFoundError:
        print(f"Ошибка: Файл {file_path} не найден!")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        sys.exit(1)


def save_ontology(ontology: dict, output_file: str):
    """Сохраняет онтологию в текстовый файл."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ОНТОЛОГИЯ СВЯЗЕЙ\n")
        f.write("=" * 80 + "\n\n")
        
        # Сущности
        f.write(f"СУЩНОСТИ (всего: {len(ontology['entities'])}):\n")
        f.write("-" * 80 + "\n")
        for name, data in ontology['entities'].items():
            f.write(f"  {name} [{data['type']}]\n")
            f.write(f"    Упоминаний: {data['attributes'].get('mentions', 0)}\n")
            f.write(f"    Всего связей: {data['attributes'].get('total_relations', 0)}\n")
            f.write("\n")
        
        # Связи
        f.write(f"\nСВЯЗИ (всего: {len(ontology['relations'])}):\n")
        f.write("-" * 80 + "\n")
        for relation in ontology['relations']:
            f.write(f"  {relation['source']} --[{relation['type']}]--> {relation['target']}\n")
            if relation.get('context'):
                f.write(f"    Контекст: {relation['context'][:100]}...\n")
            f.write("\n")


def main():
    """Основная функция программы."""
    print("=" * 80)
    print("АНАЛИЗ СВЯЗЕЙ ИМЁН СОБСТВЕННЫХ В КНИГЕ")
    print("=" * 80)
    print()
    
    # Проверяем аргументы командной строки
    if len(sys.argv) < 2:
        print("Использование: python main.py <путь_к_книге.txt> [путь_к_выходному_графу.html]")
        print()
        print("Пример:")
        print("  python main.py book.txt graph.html")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_graph = sys.argv[2] if len(sys.argv) > 2 else 'graph.html'
    output_ontology = 'ontology.txt'
    
    print(f"📖 Загрузка книги из файла: {input_file}")
    text = load_text_from_file(input_file)
    print(f"   Загружено символов: {len(text):,}")
    print()
    
    # Шаг 1: Извлечение сущностей
    print("🔍 Шаг 1: Извлечение имен собственных...")
    entity_extractor = EntityExtractor()
    entities = entity_extractor.extract_entities(text)
    
    total_entities = sum(len(ents) for ents in entities.values())
    print(f"   Найдено сущностей:")
    print(f"     - Персоны: {len(entities.get('PERSON', []))}")
    print(f"     - Локации: {len(entities.get('LOC', []))}")
    print(f"     - Организации: {len(entities.get('ORG', []))}")
    print(f"     - Всего: {total_entities}")
    print()
    
    # Шаг 2: Извлечение связей
    print("🔗 Шаг 2: Извлечение связей между сущностями...")
    relation_extractor = RelationExtractor()
    relations = relation_extractor.extract_relations(text, entities)
    print(f"   Найдено связей: {len(relations)}")
    
    if relations:
        relation_types = {}
        for rel in relations:
            rel_type = rel['type']
            relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
        print(f"   Типы связей:")
        for rel_type, count in relation_types.items():
            print(f"     - {rel_type}: {count}")
    print()
    
    # Шаг 3: Построение онтологии
    print("📊 Шаг 3: Построение онтологии...")
    ontology_builder = OntologyBuilder()
    ontology = ontology_builder.build_ontology(entities, relations)
    statistics = ontology_builder.get_statistics()
    print(f"   Онтология построена!")
    print(f"   Статистика:")
    print(f"     - Сущностей: {statistics['total_entities']}")
    print(f"     - Связей: {statistics['total_relations']}")
    print()
    
    # Сохранение онтологии
    print(f"💾 Сохранение онтологии в {output_ontology}...")
    save_ontology(ontology, output_ontology)
    print()
    
    # Шаг 4: Визуализация графа
    print("🎨 Шаг 4: Построение графа...")
    graph_visualizer = GraphVisualizer()
    graph = graph_visualizer.build_graph(ontology)
    graph_info = graph_visualizer.get_graph_info()
    
    print(f"   Граф построен!")
    print(f"     - Узлов: {graph_info['nodes']}")
    print(f"     - Рёбер: {graph_info['edges']}")
    print(f"     - Плотность: {graph_info['density']:.4f}")
    print(f"     - Связных компонентов: {graph_info['components']}")
    print()
    
    print(f"🎨 Создание визуализации...")
    output_path = graph_visualizer.visualize_interactive(output_file=output_graph)
    print()
    
    print("=" * 80)
    print("✅ ГОТОВО!")
    print("=" * 80)
    print(f"📊 Граф сохранен: {output_path}")
    print(f"📄 Онтология сохранена: {output_ontology}")
    print()
    print("Откройте HTML файл в браузере для просмотра интерактивного графа!")


if __name__ == '__main__':
    main()

