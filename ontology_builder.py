"""
Модуль для построения онтологии из извлеченных связей.
Создает структурированное представление знаний о сущностях и их отношениях.
Поддерживает экспорт в OWL формат для открытия в Protege.
"""
from typing import Dict, List, Set
from collections import defaultdict
import re


class OntologyBuilder:
    """Класс для построения онтологии из связей."""
    
    def __init__(self):
        """Инициализация онтологии."""
        self.ontology = {
            'entities': {},  # {name: {type: 'PERSON'|'LOC'|'ORG', attributes: {...}}}
            'relations': [],  # [{source, target, type, ...}]
            'relation_types': set()
        }
    
    def build_ontology(self, entities: Dict[str, List[Dict]], relations: List[Dict]) -> Dict:
        """
        Строит онтологию из извлеченных сущностей и связей.
        
        Args:
            entities: Словарь с сущностями
            relations: Список связей между сущностями
            
        Returns:
            Онтология в виде словаря
        """
        # Добавляем сущности
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                name = entity['normalized']
                if name not in self.ontology['entities']:
                    self.ontology['entities'][name] = {
                        'type': entity_type,
                        'attributes': {
                            'mentions': 1,
                            'first_mention': entity.get('start', 0)
                        }
                    }
                else:
                    self.ontology['entities'][name]['attributes']['mentions'] += 1
        
        # Добавляем связи
        for relation in relations:
            source = relation['source']
            target = relation['target']
            rel_type = relation['type']
            
            # Убеждаемся, что обе сущности есть в онтологии
            if source not in self.ontology['entities']:
                self.ontology['entities'][source] = {
                    'type': relation.get('source_type', 'UNKNOWN'),
                    'attributes': {'mentions': 0}
                }
            
            if target not in self.ontology['entities']:
                self.ontology['entities'][target] = {
                    'type': relation.get('target_type', 'UNKNOWN'),
                    'attributes': {'mentions': 0}
                }
            
            # Добавляем связь
            self.ontology['relations'].append({
                'source': source,
                'target': target,
                'type': rel_type,
                'confidence': relation.get('confidence', 0.5),
                'context': relation.get('context', '')
            })
            
            self.ontology['relation_types'].add(rel_type)
        
        # Вычисляем дополнительные атрибуты для сущностей
        self._enrich_entities()
        
        return self.ontology
    
    def _enrich_entities(self):
        """Обогащает сущности дополнительными атрибутами на основе связей."""
        # Считаем количество связей для каждой сущности
        relation_count = defaultdict(int)
        relation_types_count = defaultdict(lambda: defaultdict(int))
        
        for relation in self.ontology['relations']:
            source = relation['source']
            target = relation['target']
            rel_type = relation['type']
            
            relation_count[source] += 1
            relation_count[target] += 1
            relation_types_count[source][rel_type] += 1
            relation_types_count[target][rel_type] += 1
        
        # Добавляем статистику в атрибуты
        for entity_name, entity_data in self.ontology['entities'].items():
            entity_data['attributes']['total_relations'] = relation_count[entity_name]
            entity_data['attributes']['relation_types'] = dict(relation_types_count[entity_name])
    
    def get_statistics(self) -> Dict:
        """Возвращает статистику по онтологии."""
        return {
            'total_entities': len(self.ontology['entities']),
            'total_relations': len(self.ontology['relations']),
            'entity_types': {
                entity_type: sum(1 for e in self.ontology['entities'].values() if e['type'] == entity_type)
                for entity_type in ['PERSON', 'LOC', 'ORG']
            },
            'relation_types': {
                rel_type: sum(1 for r in self.ontology['relations'] if r['type'] == rel_type)
                for rel_type in self.ontology['relation_types']
            }
        }
    
    def export_to_owl(self, output_file: str, ontology_name: str = "BookConnections"):
        """
        Экспортирует онтологию в OWL формат для открытия в Protege.
        
        Args:
            output_file: Путь к выходному OWL файлу
            ontology_name: Имя онтологии
        """
        try:
            from owlready2 import (get_ontology, Thing, ObjectProperty, DatatypeProperty,
                                 FunctionalProperty, SymmetricProperty)
        except ImportError:
            raise ImportError("Для экспорта в OWL требуется библиотека owlready2. Установите: pip install owlready2")
        
        # Создаем новую онтологию
        iri = f"http://example.org/{ontology_name}.owl#"
        onto = get_ontology(iri)
        
        # Создаем базовые классы для типов сущностей
        with onto:
            class Person(Thing):
                pass
            
            class Location(Thing):
                pass
            
            class Organization(Thing):
                pass
            
            # Создаем объектные свойства для типов связей
            # Используем синтаксис Person >> Location для domain и range
            class hasResidence(ObjectProperty, FunctionalProperty):
                domain = [Person]
                range = [Location]
            
            class hasResident(ObjectProperty):
                domain = [Location]
                range = [Person]
            
            class hasFamilyRelation(ObjectProperty):
                domain = [Person]
                range = [Person]
            
            class hasFriendship(ObjectProperty, SymmetricProperty):
                domain = [Person]
                range = [Person]
            
            class hasWorkRelation(ObjectProperty):
                domain = [Person]
                range = [Person]
            
            class hasLoveRelation(ObjectProperty):
                domain = [Person]
                range = [Person]
            
            # Создаем свойства данных
            class mentions(DatatypeProperty):
                domain = [Thing]
                range = [int]
            
            class confidence(DatatypeProperty):
                domain = [ObjectProperty]
                range = [float]
        
        # Маппинг типов связей на OWL свойства
        relation_to_property = {
            'RESIDENCE': hasResidence,
            'HAS_RESIDENT': hasResident,
            'FAMILY': hasFamilyRelation,
            'FRIENDSHIP': hasFriendship,
            'WORK': hasWorkRelation,
            'LOVE': hasLoveRelation
        }
        
        # Маппинг типов сущностей на OWL классы
        entity_type_to_class = {
            'PERSON': Person,
            'LOC': Location,
            'ORG': Organization
        }
        
        # Создаем индивиды (экземпляры) для всех сущностей
        entity_instances = {}
        for entity_name, entity_data in self.ontology['entities'].items():
            entity_type = entity_data['type']
            owl_class = entity_type_to_class.get(entity_type, Thing)
            
            # Очищаем имя для OWL (убираем недопустимые символы)
            clean_name = self._clean_owl_name(entity_name)
            instance = owl_class(clean_name)
            
            # Добавляем аннотацию с оригинальным именем
            instance.label = [entity_name]
            
            # Добавляем количество упоминаний
            mentions_count = entity_data['attributes'].get('mentions', 0)
            if mentions_count > 0:
                instance.mentions = [mentions_count]
            
            entity_instances[entity_name] = instance
        
        # Создаем связи между индивидами
        for relation in self.ontology['relations']:
            source = relation['source']
            target = relation['target']
            rel_type = relation['type']
            confidence_val = relation.get('confidence', 0.5)
            
            if source not in entity_instances or target not in entity_instances:
                continue
            
            source_instance = entity_instances[source]
            target_instance = entity_instances[target]
            property_obj = relation_to_property.get(rel_type)
            
            if property_obj:
                try:
                    # Добавляем связь (в owlready2 используется синтаксис instance.property.append(value))
                    getattr(source_instance, property_obj.name).append(target_instance)
                except Exception as e:
                    # Если не удалось добавить связь, пропускаем
                    print(f"Не удалось добавить связь {source} -> {target}: {e}")
                    continue
        
        # Сохраняем онтологию в файл
        onto.save(file=output_file, format="rdfxml")
        print(f"Онтология сохранена в OWL формате: {output_file}")
        
        return output_file
    
    def _clean_owl_name(self, name: str) -> str:
        """
        Очищает имя для использования в OWL.
        Убирает недопустимые символы и заменяет пробелы.
        """
        # Заменяем пробелы на подчеркивания
        clean = name.replace(' ', '_')
        # Убираем недопустимые символы (оставляем только буквы, цифры, подчеркивания)
        clean = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9_]', '_', clean)
        # Убираем множественные подчеркивания
        clean = re.sub(r'_+', '_', clean)
        # Убираем подчеркивания в начале и конце
        clean = clean.strip('_')
        # Если имя начинается с цифры, добавляем префикс
        if clean and clean[0].isdigit():
            clean = 'entity_' + clean
        # Если пустое, используем дефолтное имя
        if not clean:
            clean = 'Entity'
        return clean

