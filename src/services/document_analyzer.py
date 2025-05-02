"""
Document Analyzer for CollabGPT.

This module provides functions for analyzing document content and changes,
including summarization, content classification, and entity extraction.
"""

import re
from typing import Dict, List, Any, Tuple, Set, Optional
import difflib
from datetime import datetime
import nltk
import logging
from enum import Enum

from ..utils.performance import measure_latency, get_performance_monitor
from ..utils import logger

# Download all required NLTK resources upfront
def ensure_nltk_resources():
    """Ensure all required NLTK resources are downloaded."""
    resources = [
        'punkt',
        'stopwords',
        'averaged_perceptron_tagger',
        'wordnet'
    ]
    
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}")
            print(f"NLTK resource already available: {resource}")
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)

# Call this immediately to ensure resources are available
ensure_nltk_resources()

# Only import what doesn't rely on punkt_tab
from nltk.corpus import stopwords
from nltk.probability import FreqDist

# Create custom tokenizers to avoid NLTK dependencies
def simple_sent_tokenize(text):
    """A simple sentence tokenizer that doesn't rely on punkt_tab."""
    if not text:
        return []
    # Split by common sentence terminators
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out empty sentences
    return [s.strip() for s in sentences if s.strip()]

def simple_word_tokenize(text):
    """A simple word tokenizer that doesn't rely on punkt_tab."""
    if not text:
        return []
    # Remove punctuation and split by whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    return [w.strip() for w in words if w.strip()]


class ChangeImportance(Enum):
    """Enum for classifying change importance levels."""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    TRIVIAL = 1
    
    @classmethod
    def get_description(cls, level):
        """Get a human-readable description of the importance level."""
        descriptions = {
            cls.CRITICAL: "Critical changes that significantly alter document meaning or structure",
            cls.HIGH: "Important changes to key sections or major content updates",
            cls.MEDIUM: "Moderate changes affecting document flow or clarity",
            cls.LOW: "Minor edits with limited impact on overall document",
            cls.TRIVIAL: "Small formatting changes or typo corrections"
        }
        return descriptions.get(level, "Unknown importance level")


class DocumentAnalyzer:
    """
    Analyzes document content and changes to provide insights and summaries.
    """
    
    def __init__(self, language: str = 'english'):
        """
        Initialize the document analyzer.
        
        Args:
            language: The language for NLP operations (default: English)
        """
        self.language = language
        try:
            self.stop_words = set(stopwords.words(language))
        except:
            print(f"Warning: Stopwords not available for language '{language}'. Using empty set.")
            self.stop_words = set()
        self.document_cache = {}
        self.logger = logger.get_logger("document_analyzer")
        self.performance_monitor = get_performance_monitor()
        
        # Keywords that might indicate important changes when found in added/removed content
        self.importance_keywords = {
            'critical': [
                'urgent', 'critical', 'deadline', 'immediately', 'emergency', 'crucial',
                'vital', 'required', 'must', 'critical', 'warning', 'danger', 'alert'
            ],
            'high': [
                'important', 'significant', 'major', 'key', 'essential', 'primary', 
                'fundamental', 'core', 'central', 'priority', 'necessary'
            ],
            'medium': [
                'update', 'modify', 'change', 'revise', 'enhance', 'improve', 'adjust',
                'develop', 'expand', 'extend', 'clarify', 'detail'
            ]
        }
    
    def analyze_document(self, document_id: str, content: str) -> Dict[str, Any]:
        """
        Perform full analysis of a document's content.
        
        Args:
            document_id: The document identifier
            content: The full text content of the document
            
        Returns:
            Dictionary containing analysis results
        """
        with measure_latency("document_analysis", self.performance_monitor):
            # Store content in cache
            self.document_cache[document_id] = {
                'content': content,
                'timestamp': datetime.now(),
                'analysis': {}
            }
            
            # Perform analysis
            analysis = {
                'summary': self.summarize_text(content),
                'word_count': len(simple_word_tokenize(content)),
                'sentence_count': len(simple_sent_tokenize(content)),
                'key_phrases': self._extract_key_phrases(content),
                'sections': self._identify_sections(content),
                'language_metrics': self._analyze_language(content),
                'entities': self._extract_entities(content)
            }
            
            # Update cache with analysis
            self.document_cache[document_id]['analysis'] = analysis
            
            return analysis
    
    def analyze_changes(self, document_id: str, previous_content: str, current_content: str) -> Dict[str, Any]:
        """
        Analyze changes between two versions of a document.
        
        Args:
            document_id: The document identifier
            previous_content: The previous version of the document
            current_content: The current version of the document
            
        Returns:
            Dictionary containing change analysis
        """
        with measure_latency("change_analysis", self.performance_monitor):
            # Get diff between versions
            diff = self._get_diff(previous_content, current_content)
            
            # Extract additions and deletions
            additions = ''.join([chunk for tag, chunk in diff if tag == 1])
            deletions = ''.join([chunk for tag, chunk in diff if tag == -1])
            
            # Identify changed sections
            changed_sections = self._identify_changed_sections(previous_content, current_content)
            
            # Categorize the importance of changes
            importance_level, importance_reasons = self._categorize_change_importance(
                additions, deletions, changed_sections
            )
            
            # Analyze the changes
            change_analysis = {
                'document_id': document_id,
                'timestamp': datetime.now(),
                'changes': {
                    'added_content': additions,
                    'deleted_content': deletions,
                    'added_word_count': len(simple_word_tokenize(additions)) if additions else 0,
                    'deleted_word_count': len(simple_word_tokenize(deletions)) if deletions else 0,
                    'changed_sections': changed_sections,
                    'change_summary': self.summarize_changes(previous_content, current_content),
                    'importance': {
                        'level': importance_level.name,
                        'level_value': importance_level.value,
                        'reasons': importance_reasons,
                        'description': ChangeImportance.get_description(importance_level)
                    }
                }
            }
            
            return change_analysis
    
    def summarize_text(self, text: str, max_sentences: int = 3) -> str:
        """
        Generate a concise summary of text content.
        
        Args:
            text: The text to summarize
            max_sentences: Maximum number of sentences in summary
            
        Returns:
            Summarized text
        """
        if not text:
            return ""
            
        # Tokenize the text into sentences using our simple tokenizer
        sentences = simple_sent_tokenize(text)
        
        if len(sentences) <= max_sentences:
            return text
            
        # Clean and tokenize the text
        words = [word.lower() for word in simple_word_tokenize(text) if word.isalnum()]
        
        # Remove stop words
        filtered_words = [word for word in words if word not in self.stop_words]
        
        # Calculate word frequencies
        word_frequencies = FreqDist(filtered_words)
        
        # Calculate sentence scores based on word frequencies
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            for word in simple_word_tokenize(sentence.lower()):
                if word in word_frequencies:
                    if i not in sentence_scores:
                        sentence_scores[i] = 0
                    sentence_scores[i] += word_frequencies[word]
        
        # Get the top sentences
        top_sentence_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:max_sentences]
        top_sentence_indices.sort()  # Sort to maintain original order
        
        # Combine the top sentences
        summary = ' '.join([sentences[i] for i in top_sentence_indices])
        
        return summary
    
    def summarize_changes(self, previous_content: str, current_content: str) -> str:
        """
        Generate a human-readable summary of changes between document versions.
        
        Args:
            previous_content: The previous version of the document
            current_content: The current version of the document
            
        Returns:
            Summary of changes
        """
        if not previous_content and current_content:
            return "Document created with initial content."
        elif previous_content and not current_content:
            return "All document content was removed."
        elif not previous_content and not current_content:
            return "No changes detected (document is empty)."
        
        # Use the new categorized change analysis
        categories = self.categorize_changes_by_type(previous_content, current_content)
        
        # Build a more intelligent summary using the categories
        summary_parts = []
        
        # Major revisions take precedence
        if categories['major_revisions']:
            count = len(categories['major_revisions'])
            sample = self.summarize_text(categories['major_revisions'][0], 1) if categories['major_revisions'] else ""
            summary_parts.append(f"Major revision{'s' if count > 1 else ''} with significant content changes" + 
                               (f", including: '{sample}...'" if sample else "."))
        
        # Structural changes are important
        if categories['structural_changes']:
            summary_parts.append(f"Document structure was modified: " + 
                               f"{'; '.join(categories['structural_changes'][:2])}" + 
                               (f" and {len(categories['structural_changes'])-2} more changes" if len(categories['structural_changes']) > 2 else ""))
        
        # Additions and deletions
        if categories['content_additions'] and not categories['major_revisions']:
            added_text = ' '.join(categories['content_additions'])
            added_words = len(simple_word_tokenize(added_text))
            if added_words > 50:
                summary = self.summarize_text(added_text, 1)
                summary_parts.append(f"Added {added_words} words of new content" + (f", including: '{summary}'" if summary else "."))
            else:
                summary_parts.append(f"Added: '{added_text[:100]}{'...' if len(added_text) > 100 else ''}'")
        
        if categories['content_deletions'] and not categories['major_revisions']:
            deleted_text = ' '.join(categories['content_deletions'])
            deleted_words = len(simple_word_tokenize(deleted_text))
            if deleted_words > 50:
                summary = self.summarize_text(deleted_text, 1)
                summary_parts.append(f"Removed {deleted_words} words of content" + (f", including: '{summary}'" if summary else "."))
            else:
                summary_parts.append(f"Removed: '{deleted_text[:100]}{'...' if len(deleted_text) > 100 else ''}'")
        
        # Corrections
        if categories['corrections']:
            if len(categories['corrections']) == 1:
                correction = categories['corrections'][0]
                summary_parts.append(f"Made a correction: '{correction['before'][:20]}...' to '{correction['after'][:20]}...'")
            else:
                summary_parts.append(f"Made {len(categories['corrections'])} text corrections throughout the document.")
        
        # Formatting changes
        if categories['formatting_changes'] and not summary_parts:
            summary_parts.append(f"Made {len(categories['formatting_changes'])} formatting or style changes.")
        
        if not summary_parts:
            return "Minor changes made with no significant content additions or removals."
        
        return " ".join(summary_parts)
    
    def _get_diff(self, text1: str, text2: str) -> List[Tuple[int, str]]:
        """
        Get the differences between two texts.
        
        Args:
            text1: First text (previous)
            text2: Second text (current)
            
        Returns:
            List of (tag, chunk) tuples where tag is -1 for deletion, 0 for equal, 1 for addition
        """
        # Split texts into sentences for better diff readability
        lines1 = simple_sent_tokenize(text1)
        lines2 = simple_sent_tokenize(text2)
        
        # Generate the diff
        differ = difflib.SequenceMatcher(None, lines1, lines2)
        
        # Format the output
        result = []
        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            if tag == 'replace':
                result.append((-1, ' '.join(lines1[i1:i2])))
                result.append((1, ' '.join(lines2[j1:j2])))
            elif tag == 'delete':
                result.append((-1, ' '.join(lines1[i1:i2])))
            elif tag == 'insert':
                result.append((1, ' '.join(lines2[j1:j2])))
            elif tag == 'equal':
                result.append((0, ' '.join(lines1[i1:i2])))
                
        return result
    
    def _extract_key_phrases(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract key phrases from document text.
        
        Args:
            text: The document text
            top_n: Number of key phrases to extract
            
        Returns:
            List of key phrases
        """
        # A simple implementation based on n-grams and frequency
        words = [word.lower() for word in simple_word_tokenize(text) if word.isalnum()]
        filtered_words = [word for word in words if word not in self.stop_words]
        
        # Create bigrams and trigrams
        bigrams = [' '.join(filtered_words[i:i+2]) for i in range(len(filtered_words)-1)]
        trigrams = [' '.join(filtered_words[i:i+3]) for i in range(len(filtered_words)-2)]
        
        # Count frequencies
        phrases = filtered_words + bigrams + trigrams
        freq_dist = FreqDist(phrases)
        
        # Return top phrases
        return [phrase for phrase, _ in freq_dist.most_common(top_n)]
    
    def _identify_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify document sections and their content.
        
        Args:
            text: The document text
            
        Returns:
            List of section information
        """
        # Use heuristics to identify section headings
        lines = text.split('\n')
        sections = []
        current_section = None
        current_content = []
        
        heading_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown headings
            r'^([A-Z][A-Za-z\s]+)$',  # All caps or title case, standalone
            r'^([IVX]+\.\s+.+)$',  # Roman numeral headings
            r'^(\d+\.\s+.+)$'  # Numbered headings
        ]
        
        for i, line in enumerate(lines):
            line = line.strip()
            is_heading = False
            
            for pattern in heading_patterns:
                match = re.match(pattern, line)
                if match:
                    # Save previous section if exists
                    if current_section:
                        sections.append({
                            'title': current_section,
                            'content': '\n'.join(current_content),
                            'level': 1,  # Simplified level detection
                            'position': len(sections)
                        })
                    
                    # Start new section
                    current_section = match.group(1)
                    current_content = []
                    is_heading = True
                    break
            
            if not is_heading and line:
                current_content.append(line)
        
        # Add the final section
        if current_section:
            sections.append({
                'title': current_section,
                'content': '\n'.join(current_content),
                'level': 1,
                'position': len(sections)
            })
        elif current_content:  # Content without headings
            sections.append({
                'title': 'Untitled Section',
                'content': '\n'.join(current_content),
                'level': 1,
                'position': 0
            })
            
        return sections
    
    def _identify_changed_sections(self, previous_content: str, current_content: str) -> List[Dict[str, Any]]:
        """
        Identify which document sections have changed.
        
        Args:
            previous_content: The previous version of the document
            current_content: The current version of the document
            
        Returns:
            List of changed section information
        """
        prev_sections = self._identify_sections(previous_content)
        curr_sections = self._identify_sections(current_content)
        
        changed_sections = []
        
        # Check for new sections
        prev_titles = {section['title'] for section in prev_sections}
        for section in curr_sections:
            if section['title'] not in prev_titles:
                changed_sections.append({
                    'title': section['title'],
                    'change_type': 'added',
                    'content': section['content']
                })
                continue
            
            # Check for modified sections
            for prev_section in prev_sections:
                if prev_section['title'] == section['title']:
                    if prev_section['content'] != section['content']:
                        changed_sections.append({
                            'title': section['title'],
                            'change_type': 'modified',
                            'previous_content': prev_section['content'],
                            'current_content': section['content']
                        })
                    break
        
        # Check for removed sections
        curr_titles = {section['title'] for section in curr_sections}
        for section in prev_sections:
            if section['title'] not in curr_titles:
                changed_sections.append({
                    'title': section['title'],
                    'change_type': 'removed',
                    'content': section['content']
                })
                
        return changed_sections
    
    def _analyze_language(self, text: str) -> Dict[str, Any]:
        """
        Analyze language characteristics of the text.
        
        Args:
            text: The document text
            
        Returns:
            Dictionary of language metrics
        """
        words = simple_word_tokenize(text)
        sentences = simple_sent_tokenize(text)
        
        # Calculate metrics
        word_count = len(words)
        sentence_count = len(sentences)
        avg_words_per_sentence = word_count / max(sentence_count, 1)
        
        # Calculate vocabulary richness
        unique_words = set(word.lower() for word in words if word.isalnum())
        vocabulary_richness = len(unique_words) / max(word_count, 1)
        
        # Calculate readability using simplified formula (based on words per sentence)
        readability_score = 206.835 - (1.015 * avg_words_per_sentence)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_words_per_sentence': avg_words_per_sentence,
            'vocabulary_richness': vocabulary_richness,
            'readability_score': readability_score,
            'estimated_reading_time_minutes': word_count / 200  # Assumes 200 words per minute
        }
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Args:
            text: The document text
            
        Returns:
            Dictionary mapping entity types to lists of entities
        """
        # Note: For a more robust solution, we'd use spaCy or a similar NLP library
        # This is a simplified version using regex patterns
        
        entities = {
            'dates': [],
            'emails': [],
            'urls': [],
            'potential_names': []
        }
        
        # Extract dates (simple patterns)
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',  # MM-DD-YYYY
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',  # DD Mon YYYY
        ]
        
        for pattern in date_patterns:
            entities['dates'].extend(re.findall(pattern, text, re.IGNORECASE))
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['emails'] = re.findall(email_pattern, text)
        
        # Extract URLs
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        entities['urls'] = re.findall(url_pattern, text)
        
        # Extract potential names (simplified)
        lines = text.split('\n')
        for line in lines:
            words = line.split()
            for i in range(len(words) - 1):
                word1 = words[i].strip('.,;:()[]{}"\'"')
                word2 = words[i + 1].strip('.,;:()[]{}"\'"')
                
                if (word1 and word2 and
                    word1[0].isupper() and word2[0].isupper() and
                    word1.isalpha() and word2.isalpha() and
                    len(word1) > 1 and len(word2) > 1):
                    entities['potential_names'].append(f"{word1} {word2}")
        
        # Remove duplicates
        for entity_type in entities:
            entities[entity_type] = list(set(entities[entity_type]))
        
        return entities
    
    def _categorize_change_importance(self, additions: str, deletions: str, 
                                     changed_sections: List[Dict[str, Any]]) -> Tuple[ChangeImportance, List[str]]:
        """
        Categorize the importance of document changes.
        
        Args:
            additions: Text that was added
            deletions: Text that was removed
            changed_sections: List of changed sections
            
        Returns:
            Tuple of (importance_level, reasons_list)
        """
        reasons = []
        score = 0
        
        # Calculate the volume of changes
        added_words = len(simple_word_tokenize(additions)) if additions else 0
        deleted_words = len(simple_word_tokenize(deletions)) if deletions else 0
        total_changed_words = added_words + deleted_words
        
        # Check sections affected
        section_changes = {section['change_type']: 0 for section in changed_sections}
        for section in changed_sections:
            section_changes[section['change_type']] += 1
            
            # Check if important sections were modified
            if 'title' in section:
                lower_title = section['title'].lower()
                if any(kw in lower_title for kw in ['introduction', 'conclusion', 'summary', 'abstract', 'results']):
                    score += 2
                    reasons.append(f"Changes to important '{section['title']}' section")
        
        # Addition of new sections is significant
        if section_changes.get('added', 0) > 0:
            score += 2 * section_changes['added']
            reasons.append(f"Added {section_changes['added']} new section(s)")
            
        # Removal of sections is very significant
        if section_changes.get('removed', 0) > 0:
            score += 3 * section_changes['removed']
            reasons.append(f"Removed {section_changes['removed']} section(s)")
        
        # Check for important keywords in additions
        for importance, keywords in self.importance_keywords.items():
            for keyword in keywords:
                if f" {keyword} " in f" {additions.lower()} ":
                    if importance == 'critical':
                        score += 5
                        reasons.append(f"Critical keyword '{keyword}' in additions")
                    elif importance == 'high':
                        score += 3
                        reasons.append(f"Important keyword '{keyword}' in additions")
                    elif importance == 'medium':
                        score += 1
                        
        # Check for important keywords in deletions (more significant)
        for importance, keywords in self.importance_keywords.items():
            for keyword in keywords:
                if f" {keyword} " in f" {deletions.lower()} ":
                    if importance == 'critical':
                        score += 6
                        reasons.append(f"Critical keyword '{keyword}' in deletions")
                    elif importance == 'high':
                        score += 4
                        reasons.append(f"Important keyword '{keyword}' in deletions")
                    elif importance == 'medium':
                        score += 2
                        
        # Account for volume of changes
        if total_changed_words > 500:
            score += 5
            reasons.append(f"Major changes with {total_changed_words} words modified")
        elif total_changed_words > 200:
            score += 3
            reasons.append(f"Significant changes with {total_changed_words} words modified")
        elif total_changed_words > 50:
            score += 1
            reasons.append(f"Moderate changes with {total_changed_words} words modified")
            
        # Determine final importance level based on score
        if score >= 10:
            importance = ChangeImportance.CRITICAL
        elif score >= 7:
            importance = ChangeImportance.HIGH
        elif score >= 4:
            importance = ChangeImportance.MEDIUM
        elif score >= 2:
            importance = ChangeImportance.LOW
        else:
            importance = ChangeImportance.TRIVIAL
            if not reasons:
                reasons.append("Minor textual changes with limited impact")
                
        return importance, reasons
    
    def categorize_changes_by_type(self, previous_content: str, current_content: str) -> Dict[str, Any]:
        """
        Categorize changes by type to enable more intelligent summaries.
        
        Args:
            previous_content: The previous version of the document
            current_content: The current version of the document
            
        Returns:
            Dictionary of categorized changes
        """
        categories = {
            'content_additions': [],
            'content_deletions': [],
            'formatting_changes': [],
            'structural_changes': [],
            'corrections': [],
            'major_revisions': []
        }
        
        # Get diff between versions
        diff = self._get_diff(previous_content, current_content)
        
        # Extract additions and deletions
        additions = [(tag, chunk) for tag, chunk in diff if tag == 1]
        deletions = [(tag, chunk) for tag, chunk in diff if tag == -1]
        
        # Check for simple corrections
        for i, (_, deletion) in enumerate(deletions):
            for j, (_, addition) in enumerate(additions):
                # If they're similar, it might be a correction
                if len(deletion) > 0 and len(addition) > 0:
                    similarity = difflib.SequenceMatcher(None, deletion, addition).ratio()
                    if similarity > 0.7:  # High similarity indicates correction rather than new content
                        categories['corrections'].append({
                            'before': deletion,
                            'after': addition,
                            'similarity': similarity
                        })
                        # Mark as processed so they're not counted as pure additions/deletions
                        deletions[i] = (-1, '')
                        additions[j] = (1, '')
                        
        # Process remaining additions
        for _, content in additions:
            if content.strip():
                # Check if it's formatting or structural
                if re.search(r'[*_#]{2,}|^#{1,6}\s+', content):
                    categories['formatting_changes'].append(content)
                elif len(simple_word_tokenize(content)) > 50:
                    categories['major_revisions'].append(content)
                else:
                    categories['content_additions'].append(content)
        
        # Process remaining deletions
        for _, content in deletions:
            if content.strip():
                if re.search(r'[*_#]{2,}|^#{1,6}\s+', content):
                    categories['formatting_changes'].append(content)
                elif len(simple_word_tokenize(content)) > 50:
                    categories['major_revisions'].append(content)
                else:
                    categories['content_deletions'].append(content)
        
        # Check for structural changes
        prev_sections = self._identify_sections(previous_content)
        curr_sections = self._identify_sections(current_content)
        
        if len(prev_sections) != len(curr_sections):
            categories['structural_changes'].append(f"Document structure changed from {len(prev_sections)} to {len(curr_sections)} sections")
        
        # Compare section order and detect rearrangements
        prev_titles = [s['title'] for s in prev_sections]
        curr_titles = [s['title'] for s in curr_sections]
        
        common_titles = set(prev_titles) & set(curr_titles)
        for title in common_titles:
            prev_idx = prev_titles.index(title)
            curr_idx = curr_titles.index(title)
            if prev_idx != curr_idx:
                categories['structural_changes'].append(f"Section '{title}' moved from position {prev_idx} to {curr_idx}")
        
        return categories