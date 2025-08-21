"""
Document Processing Pipeline for Physics Educational Content
===========================================================

This module provides comprehensive multimodal document processing capabilities
specifically designed for physics educational materials. It includes:

Components:
-----------
- LaTeX equation extraction and parsing
- Physics diagram analysis using computer vision  
- Educational content classification
- Knowledge graph integration
- Multimodal document processing (PDF, DOCX, text)
- Complete pipeline orchestration

Features:
---------
- Extracts mathematical equations from various document formats
- Analyzes physics diagrams and identifies objects, vectors, concepts
- Classifies content as problems, solutions, explanations, etc.
- Maps processed content to existing physics knowledge graph
- Handles batch processing with validation and error recovery
- Provides comprehensive logging and performance monitoring

Usage:
------
    from document_processing import DocumentProcessingPipeline, PipelineConfig
    
    config = PipelineConfig(
        output_directory="./output",
        enable_graph_integration=True,
        enable_content_classification=True
    )
    
    pipeline = DocumentProcessingPipeline(config)
    result = pipeline.process_document("physics_textbook.pdf")

Example:
--------
    # Process a single physics document
    from document_processing.pipeline_orchestrator import DocumentProcessingPipeline, PipelineConfig
    
    config = PipelineConfig(output_directory="./processed_docs")
    pipeline = DocumentProcessingPipeline(config)
    
    result = pipeline.process_document("kinematics_problems.pdf")
    
    if result.success:
        print(f"Processed {len(result.processed_document.content.equations)} equations")
        print(f"Found {len(result.classified_segments)} content segments")
        print(f"Identified concepts: {result.processed_document.physics_concepts}")

Architecture:
-------------
The pipeline consists of several interconnected components:

1. **MultimodalDocumentProcessor**: Extracts content from various document formats
2. **PhysicsLatexProcessor**: Parses mathematical equations and identifies variables
3. **PhysicsDiagramAnalyzer**: Analyzes diagrams using computer vision techniques
4. **PhysicsContentClassifier**: Classifies educational content types
5. **KnowledgeGraphIntegrator**: Maps content to existing physics concepts
6. **DocumentProcessingPipeline**: Orchestrates the complete workflow

Each component can be used independently or as part of the complete pipeline.

Requirements:
-------------
- Python 3.8+
- SymPy for mathematical processing
- OpenCV for image analysis
- Neo4j for knowledge graph storage
- PyMuPDF for PDF processing
- See requirements.txt for complete dependencies

Author: Claude Code (Physics Assistant Database Analytics Specialist)
License: MIT
Version: 1.0.0
"""

# Version information
__version__ = "1.0.0"
__author__ = "Claude Code - Physics Assistant"
__license__ = "MIT"

# Import main components for easy access
from .multimodal_processor import (
    MultimodalDocumentProcessor,
    ProcessedDocument,
    ExtractedContent,
    DocumentSection
)

from .latex_processor import (
    PhysicsLatexProcessor,
    LatexEquation,
    VariableInfo
)

from .diagram_analyzer import (
    PhysicsDiagramAnalyzer,
    PhysicsDiagram,
    DiagramElement
)

from .content_classifier import (
    PhysicsContentClassifier,
    ClassificationResult,
    EducationalSegment,
    ContentType
)

from .graph_integration import (
    KnowledgeGraphIntegrator,
    GraphIntegrationResult,
    ConceptMapping
)

from .pipeline_orchestrator import (
    DocumentProcessingPipeline,
    PipelineConfig,
    ProcessingResult,
    BatchProcessingResult
)

# Main classes available at package level
__all__ = [
    # Main pipeline
    'DocumentProcessingPipeline',
    'PipelineConfig',
    'ProcessingResult',
    'BatchProcessingResult',
    
    # Individual processors
    'MultimodalDocumentProcessor',
    'PhysicsLatexProcessor', 
    'PhysicsDiagramAnalyzer',
    'PhysicsContentClassifier',
    'KnowledgeGraphIntegrator',
    
    # Data structures
    'ProcessedDocument',
    'ExtractedContent',
    'DocumentSection',
    'LatexEquation',
    'VariableInfo',
    'PhysicsDiagram',
    'DiagramElement',
    'ClassificationResult',
    'EducationalSegment',
    'ContentType',
    'GraphIntegrationResult',
    'ConceptMapping'
]

# Package metadata
__package_info__ = {
    'name': 'document_processing',
    'version': __version__,
    'description': 'Multimodal document processing pipeline for physics educational content',
    'author': __author__,
    'license': __license__,
    'python_requires': '>=3.8',
    'keywords': ['physics', 'education', 'document processing', 'knowledge graph', 'latex', 'computer vision'],
    'classifiers': [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Education',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Text Processing :: Markup :: LaTeX'
    ]
}

def get_version():
    """Return the current version of the document processing package."""
    return __version__

def get_package_info():
    """Return complete package information."""
    return __package_info__.copy()

def create_default_pipeline(output_dir: str = "./document_processing_output") -> DocumentProcessingPipeline:
    """
    Create a DocumentProcessingPipeline with default configuration.
    
    Args:
        output_dir: Directory for pipeline outputs
        
    Returns:
        Configured DocumentProcessingPipeline instance
    """
    config = PipelineConfig(
        output_directory=output_dir,
        enable_graph_integration=True,
        enable_content_classification=True,
        enable_diagram_analysis=True,
        enable_latex_processing=True,
        validate_results=True,
        save_intermediate_results=True
    )
    
    return DocumentProcessingPipeline(config)

# Quick start function
def process_physics_document(file_path: str, output_dir: str = None) -> ProcessingResult:
    """
    Quick function to process a single physics document with default settings.
    
    Args:
        file_path: Path to the document to process
        output_dir: Optional output directory (default: ./document_processing_output)
        
    Returns:
        ProcessingResult with complete analysis
        
    Example:
        >>> result = process_physics_document("kinematics.pdf")
        >>> if result.success:
        ...     print(f"Found {len(result.processed_document.content.equations)} equations")
    """
    if output_dir is None:
        output_dir = "./document_processing_output"
    
    pipeline = create_default_pipeline(output_dir)
    
    try:
        result = pipeline.process_document(file_path)
        return result
    finally:
        pipeline.cleanup()

# Module-level configuration
import logging

# Set up basic logging for the package
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Export quick-start function
__all__.extend(['get_version', 'get_package_info', 'create_default_pipeline', 'process_physics_document'])