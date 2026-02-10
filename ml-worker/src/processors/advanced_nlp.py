import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    pipeline
)
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class AdvancedNERProcessor:
    """
    Advanced Named Entity Recognition using transformer models
    Supports multiple entity types and relation extraction
    """
    
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        """
        Initialize NER processor
        
        Options:
        - dslim/bert-base-NER: General purpose, good balance
        - xlm-roberta-large-finetuned-conll03-english: Multilingual
        - Jean-Baptiste/roberta-large-ner-english: High accuracy
        """
        self.model_name = model_name
        self.ner_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    async def initialize(self):
        """Load NER model"""
        logger.info(f"Loading NER model: {self.model_name}")
        
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model_name,
            tokenizer=self.model_name,
            aggregation_strategy="simple",  # Merge subword tokens
            device=0 if self.device == "cuda" else -1
        )
        
        logger.info("✅ NER model loaded successfully")
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text
        
        Returns:
            List of entities with type, text, score, start, end positions
        """
        if not text.strip():
            return []
        
        # Run NER
        entities = self.ner_pipeline(text)
        
        # Format results
        formatted_entities = []
        for entity in entities:
            formatted_entities.append({
                "text": entity["word"],
                "entity_type": entity["entity_group"],
                "confidence": float(entity["score"]),
                "start_char": entity["start"],
                "end_char": entity["end"]
            })
        
        return formatted_entities
    
    def extract_entities_from_segments(
        self, 
        hierarchical_segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract entities from all video segments with temporal information
        
        Returns:
            Knowledge graph with entities, their frequencies, and timestamps
        """
        knowledge_graph = {
            "entities": {},  # entity_type -> list of entities
            "entity_timeline": {},  # entity_text -> list of timestamps
            "entity_cooccurrence": {},  # Track which entities appear together
            "total_entities_found": 0
        }
        
        for segment in hierarchical_segments:
            text = segment.get("text", "")
            if not text:
                continue
            
            # Extract entities
            entities = self.extract_entities(text)
            
            # Group by type
            for entity in entities:
                entity_type = entity["entity_type"]
                entity_text = entity["text"]
                
                # Initialize type if needed
                if entity_type not in knowledge_graph["entities"]:
                    knowledge_graph["entities"][entity_type] = []
                
                # Add entity with segment context
                entity_with_context = {
                    **entity,
                    "segment_id": segment["segment_id"],
                    "timestamp": segment["start"],
                    "scene_id": segment["scene_id"]
                }
                knowledge_graph["entities"][entity_type].append(entity_with_context)
                
                # Track timeline
                if entity_text not in knowledge_graph["entity_timeline"]:
                    knowledge_graph["entity_timeline"][entity_text] = []
                knowledge_graph["entity_timeline"][entity_text].append({
                    "timestamp": segment["start"],
                    "segment_id": segment["segment_id"],
                    "confidence": entity["confidence"]
                })
                
                knowledge_graph["total_entities_found"] += 1
            
            # Track entity co-occurrence in this segment
            segment_entities = [e["text"] for e in entities]
            for i, entity1 in enumerate(segment_entities):
                for entity2 in segment_entities[i+1:]:
                    cooc_key = tuple(sorted([entity1, entity2]))
                    if cooc_key not in knowledge_graph["entity_cooccurrence"]:
                        knowledge_graph["entity_cooccurrence"][cooc_key] = []
                    knowledge_graph["entity_cooccurrence"][cooc_key].append({
                        "segment_id": segment["segment_id"],
                        "timestamp": segment["start"]
                    })
        
        # Calculate entity frequencies and importance
        knowledge_graph["entity_statistics"] = self._calculate_entity_statistics(
            knowledge_graph
        )
        
        return knowledge_graph
    
    def _calculate_entity_statistics(self, knowledge_graph: Dict) -> Dict:
        """Calculate statistics about entities"""
        stats = {
            "entities_by_type_count": {},
            "most_frequent_entities": [],
            "entities_with_multiple_occurrences": []
        }
        
        # Count by type
        for entity_type, entities in knowledge_graph["entities"].items():
            stats["entities_by_type_count"][entity_type] = len(entities)
        
        # Find most frequent
        from collections import Counter
        entity_freq = Counter([
            e["text"] 
            for entities in knowledge_graph["entities"].values() 
            for e in entities
        ])
        stats["most_frequent_entities"] = [
            {"entity": entity, "count": count}
            for entity, count in entity_freq.most_common(20)
        ]
        
        # Entities appearing multiple times (important entities)
        stats["entities_with_multiple_occurrences"] = [
            {"entity": entity, "count": count}
            for entity, count in entity_freq.items()
            if count > 1
        ]
        
        return stats


class RelationExtractor:
    """
    Extract relationships between entities using transformer models
    """
    
    def __init__(self, model_name: str = "Babelscape/rebel-large"):
        """
        Initialize relation extractor
        
        REBEL (Relation Extraction By End-to-end Language generation)
        is specifically designed for relation extraction
        """
        self.model_name = model_name
        self.relation_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    async def initialize(self):
        """Load relation extraction model"""
        logger.info(f"Loading relation extraction model: {self.model_name}")
        
        from transformers import AutoModelForSeq2SeqLM
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        if self.device == "cuda":
            self.model = self.model.to(self.device)
        
        logger.info("✅ Relation extraction model loaded")
    
    def extract_relations(
        self, 
        text: str, 
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities in text
        
        Returns:
            List of relations: {subject, predicate, object, confidence}
        """
        if not text or len(entities) < 2:
            return []
        
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate relations
        outputs = self.model.generate(
            **inputs,
            max_length=256,
            num_beams=5,
            num_return_sequences=5
        )
        
        # Decode relations
        relations = []
        for output in outputs:
            relation_text = self.tokenizer.decode(output, skip_special_tokens=True)
            parsed_relation = self._parse_relation_text(relation_text)
            if parsed_relation:
                relations.append(parsed_relation)
        
        return relations
    
    def _parse_relation_text(self, relation_text: str) -> Dict[str, Any]:
        """Parse REBEL output format"""
        # REBEL outputs in format: <triplet> subject <subj> predicate <obj> object
        import re
        
        pattern = r'<triplet>\s*([^<]+)\s*<subj>\s*([^<]+)\s*<obj>\s*([^<]+)'
        match = re.search(pattern, relation_text)
        
        if match:
            return {
                "subject": match.group(1).strip(),
                "predicate": match.group(2).strip(),
                "object": match.group(3).strip(),
                "confidence": 0.8  # REBEL doesn't provide confidence, use default
            }
        
        return None