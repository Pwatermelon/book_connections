"""
Модуль для извлечения имен собственных из текста.
Использует Natasha для распознавания персон и локаций.
Использует pymorphy2 для правильной нормализации падежей.
Использует transformers с русской моделью как резервный метод.
"""
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger,
    Doc
)
from pymorphy2 import MorphAnalyzer
from typing import List, Dict, Set, Optional
import re

# Попытка импортировать transformers для более мощного NER
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("ВНИМАНИЕ: transformers не установлен. Для лучшего поиска персон установите: pip install transformers torch")


class EntityExtractor:
    """Класс для извлечения имен собственных из текста."""
    
    def __init__(self):
        """Инициализация компонентов Natasha, pymorphy2 и transformers."""
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)
        # Pymorphy2 для нормализации падежей
        self.morph = MorphAnalyzer()
        
        # Трансформеры для более мощного NER (ленивая загрузка)
        self.ner_pipeline = None
        if TRANSFORMERS_AVAILABLE:
            try:
                # Используем русскую модель для NER
                # Модель может быть большой, загружаем только при необходимости
                self.ner_pipeline = None  # Ленивая загрузка
                self.use_transformers = True
            except Exception as e:
                print(f"Не удалось инициализировать transformers: {e}")
                self.use_transformers = False
        else:
            self.use_transformers = False
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict]]:
        """
        Извлекает имена собственные из текста.
        
        Args:
            text: Исходный текст книги
            
        Returns:
            Словарь с категориями сущностей:
            {
                'PERSON': [{'text': 'Иван', 'start': 10, 'end': 14, 'normalized': 'Иван'}, ...],
                'LOC': [{'text': 'Саратов', 'start': 50, 'end': 57, 'normalized': 'Саратов'}, ...],
                ...
            }
        """
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        doc.tag_ner(self.ner_tagger)
        
        # Нормализуем токены для получения именительного падежа
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        
        entities = {
            'PERSON': [],
            'LOC': [],
            'ORG': []
        }
        
        # Извлечение найденных сущностей из Natasha
        for span in doc.spans:
            entity_type = span.type
            if entity_type in entities:
                # Пытаемся получить нормализованную форму (лемму) из токенов
                normalized_text = self._get_normalized_form(span, doc)
                
                entities[entity_type].append({
                    'text': span.text,
                    'start': span.start,
                    'end': span.stop,
                    'normalized': normalized_text,
                    'chunks': span.chunks if hasattr(span, 'chunks') else []
                })
        
        # Отладочный вывод для диагностики
        all_spans_list = list(doc.spans)
        if len(entities['PERSON']) == 0:
            print(f"DEBUG: Natasha не нашел персон. Всего spans: {len(all_spans_list)}")
            print(f"DEBUG: Типы найденных сущностей: {[s.type for s in all_spans_list[:20]]}")
            print("DEBUG: Пробую альтернативные методы...")
        
        # Если Natasha не нашел персон, используем альтернативные методы
        if len(entities['PERSON']) == 0:
            # Сначала пробуем через морфологию
            persons_morph = self._extract_persons_with_morphology(text, doc)
            entities['PERSON'].extend(persons_morph)
            
            # Если и это не помогло, пробуем transformers
            if len(entities['PERSON']) == 0 and self.use_transformers:
                persons_transformer = self._extract_persons_with_transformers(text)
                entities['PERSON'].extend(persons_transformer)
                print(f"DEBUG: Transformers нашел персон: {len(persons_transformer)}")
        
        # Дополнительная обработка для нормализации имен
        entities['PERSON'] = self._normalize_persons(entities['PERSON'])
        entities['LOC'] = self._normalize_locations(entities['LOC'])
        
        return entities
    
    def _extract_persons_with_morphology(self, text: str, doc) -> List[Dict]:
        """
        Извлекает персон используя морфологический анализ через pymorphy2.
        Проверяет что слово действительно является именем собственным через морфологические теги.
        """
        persons = []
        
        # Получаем уже найденные позиции чтобы не дублировать
        existing_positions = set()
        for span in doc.spans:
            if span.type in ['LOC', 'ORG']:
                existing_positions.add((span.start, span.stop))
        
        # Только паттерны где имя четко в контексте действия (более строгие)
        patterns = [
            # Имя перед глаголом (субъект действия) - самое надежное
            r'([А-ЯЁA-Z][а-яёa-z]+(?:\s+[А-ЯЁA-Z][а-яёa-z]+)?)\s+(?:сказал|сказала|думал|думала|работал|работала|жил|жила|был|была|стал|стала|родился|родилась|познакомился|познакомилась|встретил|встретила)',
            # После глагола (объект) - менее надежно, но проверяем
            r'(?:сказал|сказала|встретил|встретила|познакомился|познакомилась|знал|знала|любил|любила)\s+([А-ЯЁA-Z][а-яёa-z]+(?:\s+[А-ЯЁA-Z][а-яёa-z]+)?)',
            # Имя и фамилия вместе перед глаголом
            r'([А-ЯЁA-Z][а-яёa-z]+\s+[А-ЯЁA-Z][а-яёa-z]+)\s+(?:был|была|стал|стала|жил|жила|работал|работала|родился|родилась)',
        ]
        
        found_names = set()  # Для избежания дублей
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group(1).strip()
                start, end = match.span(1)
                
                # Пропускаем если уже найдено
                name_key = name.lower()
                if name_key in found_names:
                    continue
                
                # Проверяем что не попало на уже найденную локацию
                if any(start >= pos[0] and end <= pos[1] for pos in existing_positions):
                    continue
                
                # Строгая проверка через pymorphy2 - должно быть имя собственное
                name_parts = name.split()
                is_valid_person = False
                normalized_parts = []
                
                for word in name_parts:
                    if len(word) < 2 or len(word) > 25:  # Имена обычно 2-25 символов
                        break
                    
                    try:
                        # Парсим слово через морфологию
                        parsed_list = self.morph.parse(word)
                        if not parsed_list:
                            break
                        
                        parsed = parsed_list[0]
                        tag = parsed.tag
                        
                        # Строгая проверка - должно быть имя (Name) или фамилия (Surn)
                        # или одушевленное существительное в именительном падеже
                        is_name = False
                        
                        # Проверяем теги морфологии
                        if hasattr(tag, 'PNCT'):  # Пропускаем знаки препинания
                            break
                        
                        # Ищем тег Name (имя) или Surn (фамилия) в тегах
                        tag_str = str(tag)
                        if 'Name' in tag_str or 'Surn' in tag_str:
                            is_name = True
                        elif 'NOUN' in tag_str:
                            # Может быть одушевленное существительное (имя)
                            if 'anim' in tag_str and 'nomn' in tag_str:  # одушевленное, именительный
                                # Дополнительная проверка - не должно быть обычным существительным
                                # Имена обычно не склоняются как обычные слова
                                is_name = True
                        
                        if is_name:
                            # Нормализуем
                            norm = parsed.normal_form
                            if norm:
                                # Сохраняем регистр
                                if word[0].isupper():
                                    norm = norm[0].upper() + norm[1:] if len(norm) > 1 else norm.upper()
                                normalized_parts.append(norm)
                            else:
                                normalized_parts.append(word)
                        else:
                            # Если хотя бы одно слово не имя - вся фраза не имя
                            break
                    
                    except Exception:
                        break
                
                # Если все части прошли проверку
                if len(normalized_parts) == len(name_parts) and len(normalized_parts) > 0:
                    is_valid_person = True
                    normalized = ' '.join(normalized_parts)
                
                if is_valid_person:
                    found_names.add(name_key)
                    persons.append({
                        'text': name,
                        'start': start,
                        'end': end,
                        'normalized': normalized,
                        'chunks': []
                    })
        
        return persons
    
    def _extract_persons_with_transformers(self, text: str) -> List[Dict]:
        """
        Извлекает персон используя transformers с русской моделью.
        Использует модель из huggingface для русского NER.
        """
        persons = []
        
        if not TRANSFORMERS_AVAILABLE or not self.use_transformers:
            return persons
        
        try:
            # Ленивая загрузка модели
            if self.ner_pipeline is None:
                # Используем легкую модель для русского NER
                # Можно использовать другие модели, но эта быстрая
                model_name = "surdan/rubert-base-ner"  # Популярная модель для русского NER
                try:
                    self.ner_pipeline = pipeline(
                        "ner",
                        model=model_name,
                        tokenizer=model_name,
                        aggregation_strategy="simple"
                    )
                except Exception as e:
                    print(f"Не удалось загрузить модель {model_name}: {e}")
                    print("Пробую альтернативную модель...")
                    # Альтернатива - используем другую модель или отключаем
                    self.use_transformers = False
                    return persons
            
            # Разбиваем текст на части если он слишком длинный
            max_length = 512  # Максимальная длина для большинства моделей
            chunks = []
            if len(text) > max_length:
                # Разбиваем по предложениям
                sentences = re.split(r'[.!?]\s+', text)
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < max_length:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence + ". "
                if current_chunk:
                    chunks.append(current_chunk)
            else:
                chunks = [text]
            
            # Обрабатываем каждый chunk
            for chunk in chunks:
                try:
                    results = self.ner_pipeline(chunk)
                    for entity in results:
                        if entity.get('entity_group') == 'PER' or 'PER' in str(entity.get('entity_group', '')):
                            # Найдена персона
                            persons.append({
                                'text': entity['word'],
                                'start': entity.get('start', 0),
                                'end': entity.get('end', 0),
                                'normalized': entity['word'],  # Будет нормализовано позже
                                'chunks': []
                            })
                except Exception as e:
                    print(f"Ошибка при обработке chunk через transformers: {e}")
                    continue
        
        except Exception as e:
            print(f"Ошибка при использовании transformers для NER: {e}")
            self.use_transformers = False
        
        return persons
    
    def _get_normalized_form(self, span, doc) -> str:
        """Получает нормализованную форму (именительный падеж) используя pymorphy2."""
        text = span.text.strip()
        if not text:
            return text
        
        # Используем pymorphy2 для нормализации вместо Natasha лемм
        # Разбиваем на слова
        words = text.split()
        normalized_parts = []
        
        for word in words:
            # Пытаемся получить нормальную форму через pymorphy2
            try:
                parsed = self.morph.parse(word)[0]
                normal_form = parsed.normal_form
                # Сохраняем заглавную букву если была в оригинале
                if word and word[0].isupper():
                    normal_form = normal_form[0].upper() + normal_form[1:] if len(normal_form) > 1 else normal_form.upper()
                normalized_parts.append(normal_form)
            except Exception:
                # Если не удалось нормализовать, оставляем как есть
                normalized_parts.append(word)
        
        return ' '.join(normalized_parts)
    
    def _normalize_persons(self, persons: List[Dict]) -> List[Dict]:
        """Нормализует имена персон (убирает дубликаты с похожими именами)."""
        # Группируем персон по нормализованному имени
        person_groups = {}
        
        for person in persons:
            normalized_name = person.get('normalized', person['text'].strip())
            # Приводим к единому формату: берем первое и последнее слово
            parts = normalized_name.split()
            if len(parts) >= 2:
                key = f"{parts[0]} {parts[-1]}"
            else:
                key = parts[0] if parts else normalized_name
            
            key_lower = key.lower()
            
            # Группируем: используем наиболее частое написание
            if key_lower not in person_groups:
                person_groups[key_lower] = {
                    'name': key,  # Имя в именительном падеже
                    'persons': []
                }
            
            person_groups[key_lower]['persons'].append(person)
        
        # Возвращаем уникальные персоны с нормализованным именем
        normalized = []
        for key_lower, group in person_groups.items():
            # Берем первого персонажа из группы
            person = group['persons'][0].copy()
            person['normalized'] = group['name']
            # Обновляем количество упоминаний
            person['mentions_count'] = len(group['persons'])
            normalized.append(person)
        
        return normalized
    
    def _normalize_locations(self, locations: List[Dict]) -> List[Dict]:
        """Нормализует названия локаций, объединяя разные падежи в одно имя."""
        # Группируем локации по базовому названию
        loc_groups = {}
        
        for loc in locations:
            name = loc.get('normalized', loc['text'].strip())
            original = loc['text'].strip()
            
            # Пытаемся получить базовую форму (именительный падеж)
            base_name = self._get_location_base_form(name)
            base_key = base_name.lower()
            
            if base_key not in loc_groups:
                loc_groups[base_key] = {
                    'name': base_name,  # Именительный падеж
                    'locations': []
                }
            
            loc_groups[base_key]['locations'].append(loc)
        
        # Возвращаем уникальные локации с нормализованным именем
        normalized = []
        for base_key, group in loc_groups.items():
            # Берем первую локацию из группы
            loc = group['locations'][0].copy()
            loc['normalized'] = group['name']
            # Обновляем количество упоминаний
            loc['mentions_count'] = len(group['locations'])
            normalized.append(loc)
        
        return normalized
    
    def _get_location_base_form(self, name: str) -> str:
        """Приводит название локации к именительному падежу используя pymorphy2."""
        name = name.strip()
        if not name:
            return name
        
        # Используем pymorphy2 для нормализации
        # Разбиваем на слова (для составных названий типа "Санкт-Петербург")
        words = re.split(r'[\s-]+', name)
        normalized_words = []
        
        for word in words:
            if not word:
                continue
            # Получаем нормальную форму (именительный падеж, единственное число)
            parsed = self.morph.parse(word)[0]
            normal_form = parsed.normal_form
            # Сохраняем заглавную букву если была
            if word and word[0].isupper():
                normal_form = normal_form[0].upper() + normal_form[1:] if len(normal_form) > 1 else normal_form.upper()
            normalized_words.append(normal_form)
        
        result = '-'.join(normalized_words) if '-' in name else ' '.join(normalized_words)
        return result

