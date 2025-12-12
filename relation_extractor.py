"""
Модуль для извлечения связей между сущностями.
Определяет тип связи между персонами и локациями.
"""
from typing import List, Dict, Tuple, Set
import re


class RelationExtractor:
    """Класс для извлечения связей между сущностями."""
    
    # Ключевые слова для определения связей (более гибкий подход)
    RESIDENCE_KEYWORDS = [
        'из', 'в', 'живёт', 'живет', 'проживает', 'родом', 'родился', 'родилась', 
        'уроженец', 'уроженка', 'жил', 'жила', 'жили', 'живут', 'находится',
        'from', 'lives in', 'born in', 'from', 'resides in', 'native of'
    ]
    
    FAMILY_KEYWORDS = [
        'брат', 'сестра', 'мать', 'отец', 'сын', 'дочь', 'родственники', 'родственник',
        'семья', 'родители', 'дед', 'бабушка', 'внук', 'внучка', 'дядя', 'тётя',
        'brother', 'sister', 'mother', 'father', 'son', 'daughter', 'family', 'relative',
        'parent', 'grandfather', 'grandmother', 'uncle', 'aunt'
    ]
    
    FRIENDSHIP_KEYWORDS = [
        'друзья', 'друг', 'подруга', 'знаком', 'встретил', 'встретила', 'знал', 'знала',
        'friends', 'friend', 'met', 'know', 'knew', 'acquaintance'
    ]
    
    WORK_KEYWORDS = [
        'коллеги', 'коллега', 'работал', 'работала', 'начальник', 'подчинённый',
        'colleagues', 'colleague', 'worked', 'boss', 'employee', 'manager'
    ]
    
    LOVE_KEYWORDS = [
        'любовь', 'любит', 'любила', 'женат', 'замужем', 'жена', 'муж', 'влюблён',
        'love', 'loves', 'married', 'wife', 'husband', 'in love'
    ]
    
    def __init__(self):
        """Инициализация ключевых слов."""
        pass
    
    def extract_relations(self, text: str, entities: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Извлекает связи между сущностями из текста.
        Создает двусторонние связи для всех типов отношений.
        
        Args:
            text: Исходный текст
            entities: Словарь с извлеченными сущностями
            
        Returns:
            Список связей (включая обратные):
            [{
                'source': 'Иван',
                'target': 'Саратов',
                'type': 'RESIDENCE',
                'confidence': 0.8,
                'context': 'Иван из Саратова'
            }, {
                'source': 'Саратов',
                'target': 'Иван',
                'type': 'HAS_RESIDENT',
                'confidence': 0.8,
                'context': 'Иван из Саратова'
            }, ...]
        """
        relations = []
        
        # Извлечение связей персона-локация (место жительства)
        person_loc_rels = self._extract_person_location_relations(text, entities)
        relations.extend(person_loc_rels)
        
        # Извлечение связей персона-персона
        person_person_rels = self._extract_person_person_relations(text, entities)
        relations.extend(person_person_rels)
        
        # Если не нашли связей через паттерны, пробуем более простой метод
        if len(relations) == 0 and len(entities.get('PERSON', [])) >= 2:
            relations.extend(self._extract_simple_relations(text, entities))
        
        # Создаем обратные связи для всех найденных отношений
        relations = self._create_bidirectional_relations(relations)
        
        return relations
    
    def _create_bidirectional_relations(self, relations: List[Dict]) -> List[Dict]:
        """
        Создает обратные связи для всех отношений.
        
        Типы обратных связей:
        - RESIDENCE -> HAS_RESIDENT (место имеет жителя)
        - FAMILY -> FAMILY (родство двустороннее)
        - FRIENDSHIP -> FRIENDSHIP (дружба двусторонняя)
        - WORK -> WORK (работа может быть двусторонней)
        - LOVE -> LOVE (любовь двусторонняя)
        """
        bidirectional_map = {
            'RESIDENCE': 'HAS_RESIDENT',
            'FAMILY': 'FAMILY',
            'FRIENDSHIP': 'FRIENDSHIP',
            'WORK': 'WORK',
            'LOVE': 'LOVE'
        }
        
        all_relations = list(relations)  # Копируем существующие связи
        seen_reverse = set()  # Для предотвращения дублей
        
        for relation in relations:
            rel_type = relation['type']
            source = relation['source']
            target = relation['target']
            
            # Пропускаем, если связь уже двусторонняя
            if rel_type in ['FAMILY', 'FRIENDSHIP', 'LOVE']:
                # Для симметричных связей проверяем, нет ли уже обратной
                reverse_key = (target, source, rel_type)
                if reverse_key not in seen_reverse:
                    seen_reverse.add(reverse_key)
                    # Не добавляем, если уже есть обратная связь
                    continue
            
            # Создаем обратную связь
            reverse_type = bidirectional_map.get(rel_type)
            if reverse_type:
                reverse_key = (target, source, reverse_type)
                if reverse_key not in seen_reverse:
                    seen_reverse.add(reverse_key)
                    all_relations.append({
                        'source': target,
                        'target': source,
                        'type': reverse_type,
                        'confidence': relation.get('confidence', 0.5),
                        'context': relation.get('context', ''),
                        'source_type': relation.get('target_type', 'UNKNOWN'),
                        'target_type': relation.get('source_type', 'UNKNOWN'),
                        'is_reverse': True  # Метка обратной связи
                    })
        
        return all_relations
    
    def _extract_person_location_relations(self, text: str, entities: Dict[str, List[Dict]]) -> List[Dict]:
        """Извлекает связи между персонами и локациями."""
        relations = []
        
        persons = entities.get('PERSON', [])
        locations = entities.get('LOC', [])
        
        if not persons or not locations:
            return relations
        
        # Создаем индексы для быстрого поиска
        loc_map = {loc['normalized'].lower(): loc for loc in locations}
        # Создаем словарь всех вариантов написания локаций (оригинал + все найденные варианты)
        loc_text_variants = {}
        for loc in locations:
            loc_normalized = loc['normalized'].lower()
            loc_original = loc['text'].lower()
            loc_text_variants[loc_normalized] = loc
            loc_text_variants[loc_original] = loc
            # Добавляем варианты с падежами
            base = loc_normalized.rstrip('аеиоуыэюя')
            if base and base != loc_normalized:
                for ending in ['а', 'е', 'и', 'у', 'ом', 'ой', 'е', 'ами']:
                    loc_text_variants[base + ending] = loc
        
        # Метод 1: Поиск через паттерны
        for person in persons:
            person_name = person['normalized']
            person_start = person['start']
            person_end = person['end']
            
            # Ищем контекст вокруг персоны (окно в 300 символов)
            context_start = max(0, person_start - 300)
            context_end = min(len(text), person_end + 300)
            context = text[context_start:context_end]
            context_lower = context.lower()
            
            # Метод 1: Ищем ключевые слова места жительства в контексте
            context_lower = context.lower()
            for keyword in self.RESIDENCE_KEYWORDS:
                if keyword in context_lower:
                    # Найдено ключевое слово, ищем локацию рядом
                    keyword_pos = context_lower.find(keyword)
                    # Ищем локацию в окне ±100 символов от ключевого слова
                    search_window = context[max(0, keyword_pos - 100):min(len(context), keyword_pos + 100)]
                    search_window_lower = search_window.lower()
                    
                    for loc_variant, loc_data in loc_text_variants.items():
                        # Проверяем, есть ли локация в окне поиска
                        if loc_variant in search_window_lower or any(len(part) > 3 and part in search_window_lower 
                                                                   for part in loc_variant.split()):
                            relations.append({
                                'source': person_name,
                                'target': loc_data['normalized'],
                                'type': 'RESIDENCE',
                                'confidence': 0.8,
                                'context': search_window.strip()[:150],
                                'source_type': 'PERSON',
                                'target_type': 'LOC'
                            })
                            break
        
        # Метод 2: Поиск близких упоминаний (персона и локация рядом)
        person_name_variants = {}  # Все варианты написания имен персон
        for person in persons:
            person_normalized = person['normalized']
            person_original = person['text']
            person_name_variants[person_normalized.lower()] = person
            person_name_variants[person_original.lower()] = person
            # Добавляем первое имя
            first_name = person_normalized.split()[0] if person_normalized else ''
            if first_name:
                person_name_variants[first_name.lower()] = person
        
        for person in persons:
            person_name = person['normalized']
            person_first_name = person_name.split()[0] if person_name else ''
            if not person_first_name:
                continue
                
            # Ищем все упоминания персоны (по первому имени и полному имени)
            person_positions = []
            for variant in [person_first_name, person_name]:
                if variant:
                    person_positions.extend(self._find_all_positions(text, variant))
            
            # Убираем дубликаты и сортируем
            person_positions = sorted(set(person_positions))[:20]
            
            for loc in locations:
                loc_name = loc['normalized']
                loc_original = loc['text']
                
                # Ищем все варианты написания локации
                loc_positions = []
                for loc_variant in [loc_name, loc_original]:
                    if loc_variant:
                        loc_positions.extend(self._find_all_positions(text, loc_variant))
                
                loc_positions = sorted(set(loc_positions))[:20]
                
                # Проверяем, находятся ли персона и локация рядом
                for p_pos in person_positions:
                    for l_pos in loc_positions:
                        distance = abs(p_pos - l_pos)
                        if distance < 200:  # Увеличиваем радиус поиска
                            # Берем контекст между ними
                            min_pos = min(p_pos, l_pos)
                            max_pos = max(p_pos, l_pos)
                            context = text[max(0, min_pos - 100):min(len(text), max_pos + 100)]
                            
                            # Проверяем наличие ключевых слов связи
                            context_lower = context.lower()
                            if any(keyword in context_lower for keyword in self.RESIDENCE_KEYWORDS):
                                relations.append({
                                    'source': person_name,
                                    'target': loc_name,
                                    'type': 'RESIDENCE',
                                    'confidence': 0.75,
                                    'context': context.strip()[:150],
                                    'source_type': 'PERSON',
                                    'target_type': 'LOC'
                                })
                                break
        
        # Убираем дубликаты
        seen = set()
        unique_relations = []
        for rel in relations:
            key = (rel['source'], rel['target'], rel['type'])
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)
        
        return unique_relations
    
    def _extract_person_person_relations(self, text: str, entities: Dict[str, List[Dict]]) -> List[Dict]:
        """Извлекает связи между персонами через поиск ключевых слов."""
        relations = []
        persons = entities.get('PERSON', [])
        
        if len(persons) < 2:
            return relations
        
        # Типы связей с их ключевыми словами
        relation_types_keywords = [
            (self.FAMILY_KEYWORDS, 'FAMILY'),
            (self.FRIENDSHIP_KEYWORDS, 'FRIENDSHIP'),
            (self.WORK_KEYWORDS, 'WORK'),
            (self.LOVE_KEYWORDS, 'LOVE'),
        ]
        
        # Ищем связи через ключевые слова
        for person1 in persons:
            p1_name = person1['normalized']
            p1_first = p1_name.split()[0] if p1_name else ''
            p1_original = person1['text']
            
            if not p1_first:
                continue
            
            # Ищем все упоминания первого персонажа (по всем вариантам)
            p1_positions = []
            for variant in [p1_first, p1_name, p1_original]:
                if variant:
                    p1_positions.extend(self._find_all_positions(text, variant))
            
            p1_positions = sorted(set(p1_positions))[:30]
            
            for p1_pos in p1_positions:
                # Контекст вокруг упоминания
                context_start = max(0, p1_pos - 300)
                context_end = min(len(text), p1_pos + len(p1_first) + 300)
                context = text[context_start:context_end]
                context_lower = context.lower()
                
                # Проверяем каждый тип связи через ключевые слова
                for keywords, relation_type in relation_types_keywords:
                    # Ищем ключевые слова в контексте
                    for keyword in keywords:
                        if keyword in context_lower:
                            keyword_pos = context_lower.find(keyword)
                            # Ищем второго персонажа в окне ±150 символов от ключевого слова
                            search_window = context[max(0, keyword_pos - 150):min(len(context), keyword_pos + 150)]
                            search_window_lower = search_window.lower()
                            
                            # Ищем второго персонажа в окне поиска
                            for person2 in persons:
                                if person1['normalized'] == person2['normalized']:
                                    continue
                                
                                p2_name = person2['normalized']
                                p2_variants = [p2_name.lower(), person2['text'].lower()]
                                p2_first = p2_name.split()[0] if p2_name else ''
                                if p2_first:
                                    p2_variants.append(p2_first.lower())
                                
                                if not p2_first:
                                    continue
                                
                                # Проверяем, есть ли второе имя в окне поиска
                                if any(variant in search_window_lower for variant in p2_variants if variant):
                                    # Проверяем расстояние между упоминаниями в тексте
                                    p2_positions = []
                                    for variant in [p2_first, p2_name, person2['text']]:
                                        if variant:
                                            p2_positions.extend(self._find_all_positions(text, variant))
                                    
                                    if any(abs(p1_pos - p2_pos) < 600 for p2_pos in p2_positions[:20]):
                                        relations.append({
                                            'source': p1_name,
                                            'target': p2_name,
                                            'type': relation_type,
                                            'confidence': 0.75,
                                            'context': search_window.strip()[:150],
                                            'source_type': 'PERSON',
                                            'target_type': 'PERSON'
                                        })
                                        break
        
        # Убираем дубликаты
        seen = set()
        unique_relations = []
        for rel in relations:
            key = (rel['source'], rel['target'], rel['type'])
            reverse_key = (rel['target'], rel['source'], rel['type'])
            # Не добавляем обратные связи
            if key not in seen and reverse_key not in seen:
                seen.add(key)
                unique_relations.append(rel)
        
        return unique_relations
    
    def _find_all_positions(self, text: str, substring: str) -> List[int]:
        """Находит все позиции подстроки в тексте."""
        positions = []
        start = 0
        while True:
            pos = text.lower().find(substring.lower(), start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        return positions

