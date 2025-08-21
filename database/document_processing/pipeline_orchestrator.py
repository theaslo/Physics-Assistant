#!/usr/bin/env python3
"""
Document Processing Pipeline Orchestrator
Coordinates the complete multimodal document processing workflow for physics educational content.
"""
import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
import traceback

# Import our processing components
from multimodal_processor import MultimodalDocumentProcessor, ProcessedDocument
from graph_integration import KnowledgeGraphIntegrator, GraphIntegrationResult
from content_classifier import PhysicsContentClassifier, EducationalSegment
from latex_processor import PhysicsLatexProcessor
from diagram_analyzer import PhysicsDiagramAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the document processing pipeline"""
    output_directory: str
    enable_graph_integration: bool = True
    enable_content_classification: bool = True
    enable_diagram_analysis: bool = True
    enable_latex_processing: bool = True
    validate_results: bool = True
    save_intermediate_results: bool = True
    max_file_size_mb: int = 100
    supported_formats: List[str] = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.pdf', '.docx', '.txt', '.md']

@dataclass
class ProcessingResult:
    """Complete result of document processing pipeline"""
    success: bool
    file_path: str
    processed_document: Optional[ProcessedDocument]
    graph_integration: Optional[GraphIntegrationResult]
    classified_segments: Optional[List[EducationalSegment]]
    processing_time_seconds: float
    error_message: Optional[str] = None
    warnings: List[str] = None
    intermediate_files: Dict[str, str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.intermediate_files is None:
            self.intermediate_files = {}

@dataclass
class BatchProcessingResult:
    """Result of batch processing multiple documents"""
    total_files: int
    successful_files: int
    failed_files: int
    processing_results: List[ProcessingResult]
    total_processing_time: float
    batch_summary: Dict[str, Any]

class DocumentProcessingPipeline:
    """Orchestrates the complete document processing workflow"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.setup_output_directory()
        
        # Initialize processing components
        logger.info("Initializing document processing components...")
        
        self.multimodal_processor = MultimodalDocumentProcessor(
            output_dir=os.path.join(self.config.output_directory, "extracted_content")
        )
        
        if self.config.enable_graph_integration:
            self.graph_integrator = KnowledgeGraphIntegrator()
        else:
            self.graph_integrator = None
            
        if self.config.enable_content_classification:
            self.content_classifier = PhysicsContentClassifier()
        else:
            self.content_classifier = None
            
        # Individual processors for validation
        self.latex_processor = PhysicsLatexProcessor()
        self.diagram_analyzer = PhysicsDiagramAnalyzer()
        
        logger.info("Document processing pipeline initialized successfully")
    
    def setup_output_directory(self):
        """Setup output directory structure"""
        os.makedirs(self.config.output_directory, exist_ok=True)
        
        subdirs = [
            "extracted_content",
            "processed_documents", 
            "classification_results",
            "graph_integration",
            "validation_reports",
            "logs"
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(self.config.output_directory, subdir), exist_ok=True)
    
    def process_document(self, file_path: str) -> ProcessingResult:
        """Process a single document through the complete pipeline"""
        start_time = datetime.now()
        logger.info(f"Starting pipeline processing for: {file_path}")
        
        try:
            # Validate input file
            validation_result = self._validate_input_file(file_path)
            if not validation_result['valid']:
                return ProcessingResult(
                    success=False,
                    file_path=file_path,
                    processed_document=None,
                    graph_integration=None,
                    classified_segments=None,
                    processing_time_seconds=0.0,
                    error_message=validation_result['error'],
                    warnings=validation_result.get('warnings', [])
                )
            
            warnings = validation_result.get('warnings', [])
            intermediate_files = {}
            
            # Step 1: Multimodal document processing
            logger.info("Step 1: Processing document content...")
            processed_doc = self.multimodal_processor.process_document(file_path)
            
            if self.config.save_intermediate_results:
                doc_output_path = self.multimodal_processor.save_processed_document(processed_doc)
                intermediate_files['processed_document'] = doc_output_path
            
            # Step 2: Content classification
            classified_segments = None
            if self.config.enable_content_classification and self.content_classifier:
                logger.info("Step 2: Classifying educational content...")
                classified_segments = self.content_classifier.segment_and_classify(
                    processed_doc.content.text_content
                )
                
                if self.config.save_intermediate_results:
                    classification_path = os.path.join(
                        self.config.output_directory, 
                        "classification_results",
                        f"classification_{processed_doc.document_hash[:8]}.json"
                    )
                    self.content_classifier.export_classification_results(
                        classified_segments, classification_path
                    )
                    intermediate_files['classification'] = classification_path
            
            # Step 3: Knowledge graph integration
            graph_integration_result = None
            if self.config.enable_graph_integration and self.graph_integrator:
                logger.info("Step 3: Integrating with knowledge graph...")
                try:
                    graph_integration_result = self.graph_integrator.integrate_document(processed_doc)
                    
                    if self.config.save_intermediate_results:
                        integration_path = os.path.join(
                            self.config.output_directory,
                            "graph_integration", 
                            f"integration_{processed_doc.document_hash[:8]}.json"
                        )
                        self._save_graph_integration_result(graph_integration_result, integration_path)
                        intermediate_files['graph_integration'] = integration_path
                        
                except Exception as e:
                    logger.warning(f"Graph integration failed: {str(e)}")
                    warnings.append(f"Graph integration failed: {str(e)}")
            
            # Step 4: Validation
            validation_report = None
            if self.config.validate_results:
                logger.info("Step 4: Validating processing results...")
                validation_report = self._validate_processing_results(
                    processed_doc, classified_segments, graph_integration_result
                )
                
                if self.config.save_intermediate_results:
                    validation_path = os.path.join(
                        self.config.output_directory,
                        "validation_reports",
                        f"validation_{processed_doc.document_hash[:8]}.json"
                    )
                    with open(validation_path, 'w') as f:
                        json.dump(validation_report, f, indent=2)
                    intermediate_files['validation'] = validation_path
                
                warnings.extend(validation_report.get('warnings', []))
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Create final result
            result = ProcessingResult(
                success=True,
                file_path=file_path,
                processed_document=processed_doc,
                graph_integration=graph_integration_result,
                classified_segments=classified_segments,
                processing_time_seconds=processing_time,
                warnings=warnings,
                intermediate_files=intermediate_files
            )
            
            # Save final result
            self._save_final_result(result)
            
            logger.info(f"Pipeline processing completed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            error_message = f"Pipeline processing failed: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            
            return ProcessingResult(
                success=False,
                file_path=file_path,
                processed_document=None,
                graph_integration=None,
                classified_segments=None,
                processing_time_seconds=processing_time,
                error_message=error_message
            )
    
    def process_batch(self, file_paths: List[str], max_workers: int = 3) -> BatchProcessingResult:
        """Process multiple documents in batch"""
        logger.info(f"Starting batch processing of {len(file_paths)} documents")
        start_time = datetime.now()
        
        results = []
        successful = 0
        failed = 0
        
        # Process files (could be parallelized with ThreadPoolExecutor if needed)
        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"Processing file {i}/{len(file_paths)}: {os.path.basename(file_path)}")
            
            result = self.process_document(file_path)
            results.append(result)
            
            if result.success:
                successful += 1
            else:
                failed += 1
                
            # Log progress
            logger.info(f"Batch progress: {i}/{len(file_paths)} complete ({successful} successful, {failed} failed)")
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Generate batch summary
        batch_summary = self._generate_batch_summary(results)
        
        batch_result = BatchProcessingResult(
            total_files=len(file_paths),
            successful_files=successful,
            failed_files=failed,
            processing_results=results,
            total_processing_time=total_time,
            batch_summary=batch_summary
        )
        
        # Save batch results
        self._save_batch_results(batch_result)
        
        logger.info(f"Batch processing completed: {successful}/{len(file_paths)} successful in {total_time:.2f}s")
        return batch_result
    
    def _validate_input_file(self, file_path: str) -> Dict[str, Any]:
        """Validate input file before processing"""
        result = {'valid': True, 'warnings': []}
        
        # Check file exists
        if not os.path.exists(file_path):
            return {'valid': False, 'error': f"File not found: {file_path}"}
        
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            return {'valid': False, 'error': f"File too large: {file_size_mb:.1f}MB > {self.config.max_file_size_mb}MB"}
        
        # Check file format
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.config.supported_formats:
            return {'valid': False, 'error': f"Unsupported format: {file_ext}"}
        
        # Additional validations
        if file_size_mb > 50:
            result['warnings'].append(f"Large file size: {file_size_mb:.1f}MB may take longer to process")
        
        return result
    
    def _validate_processing_results(self, processed_doc: ProcessedDocument, 
                                   classified_segments: Optional[List[EducationalSegment]],
                                   graph_integration: Optional[GraphIntegrationResult]) -> Dict[str, Any]:
        """Validate the results of document processing"""
        validation_report = {
            'validation_timestamp': datetime.now().isoformat(),
            'overall_status': 'PASS',
            'checks': {},
            'warnings': [],
            'recommendations': []
        }
        
        # Validate processed document
        doc_validation = self._validate_processed_document(processed_doc)
        validation_report['checks']['document_processing'] = doc_validation
        if not doc_validation['status']:
            validation_report['overall_status'] = 'FAIL'
        
        # Validate content classification
        if classified_segments:
            classification_validation = self._validate_classification_results(classified_segments)
            validation_report['checks']['content_classification'] = classification_validation
            validation_report['warnings'].extend(classification_validation.get('warnings', []))
        
        # Validate graph integration
        if graph_integration:
            graph_validation = self._validate_graph_integration(graph_integration)
            validation_report['checks']['graph_integration'] = graph_validation
            validation_report['warnings'].extend(graph_validation.get('warnings', []))
        
        # Generate recommendations
        validation_report['recommendations'] = self._generate_recommendations(
            processed_doc, classified_segments, graph_integration
        )
        
        return validation_report
    
    def _validate_processed_document(self, processed_doc: ProcessedDocument) -> Dict[str, Any]:
        """Validate processed document results"""
        checks = {
            'status': True,
            'text_extracted': len(processed_doc.content.text_content) > 0,
            'equations_found': len(processed_doc.content.equations),
            'diagrams_found': len(processed_doc.content.diagrams),
            'images_extracted': len(processed_doc.content.images),
            'difficulty_assessed': processed_doc.difficulty_level in ['beginner', 'intermediate', 'advanced'],
            'concepts_identified': len(processed_doc.physics_concepts) > 0,
            'issues': []
        }
        
        # Check for issues
        if not checks['text_extracted']:
            checks['issues'].append("No text content extracted")
            checks['status'] = False
        
        if len(processed_doc.content.text_content) < 50:
            checks['issues'].append("Very little text content extracted")
        
        if checks['equations_found'] == 0 and 'problem' in processed_doc.educational_classification.get('primary_type', ''):
            checks['issues'].append("Problem document with no equations found")
        
        return checks
    
    def _validate_classification_results(self, classified_segments: List[EducationalSegment]) -> Dict[str, Any]:
        """Validate content classification results"""
        if not classified_segments:
            return {'status': False, 'error': 'No segments classified'}
        
        low_confidence_count = sum(1 for seg in classified_segments if seg.classification.confidence_score < 0.5)
        unknown_type_count = sum(1 for seg in classified_segments if seg.classification.content_type.value == 'unknown')
        
        validation = {
            'status': True,
            'total_segments': len(classified_segments),
            'low_confidence_segments': low_confidence_count,
            'unknown_type_segments': unknown_type_count,
            'average_confidence': sum(seg.classification.confidence_score for seg in classified_segments) / len(classified_segments),
            'warnings': []
        }
        
        if low_confidence_count > len(classified_segments) * 0.5:
            validation['warnings'].append(f"High number of low confidence classifications: {low_confidence_count}/{len(classified_segments)}")
        
        if unknown_type_count > len(classified_segments) * 0.3:
            validation['warnings'].append(f"High number of unknown content types: {unknown_type_count}/{len(classified_segments)}")
        
        return validation
    
    def _validate_graph_integration(self, graph_integration: GraphIntegrationResult) -> Dict[str, Any]:
        """Validate graph integration results"""
        validation = {
            'status': True,
            'document_node_created': bool(graph_integration.document_node_id),
            'content_mappings': len(graph_integration.content_mappings),
            'new_nodes_created': len(graph_integration.new_nodes_created),
            'new_relationships': len(graph_integration.new_relationships_created),
            'integration_success': graph_integration.integration_metadata.get('integration_success', False),
            'warnings': []
        }
        
        if not validation['document_node_created']:
            validation['status'] = False
            validation['warnings'].append("Failed to create document node in graph")
        
        if validation['content_mappings'] == 0:
            validation['warnings'].append("No content mapped to existing concepts")
        
        return validation
    
    def _generate_recommendations(self, processed_doc: ProcessedDocument,
                                classified_segments: Optional[List[EducationalSegment]],
                                graph_integration: Optional[GraphIntegrationResult]) -> List[str]:
        """Generate recommendations for improving processing results"""
        recommendations = []
        
        # Document processing recommendations
        if len(processed_doc.content.equations) == 0 and 'physics' in processed_doc.file_path.lower():
            recommendations.append("Consider manual review - physics document with no equations detected")
        
        if len(processed_doc.physics_concepts) < 3:
            recommendations.append("Limited physics concepts identified - document may need domain expert review")
        
        # Classification recommendations
        if classified_segments:
            avg_confidence = sum(seg.classification.confidence_score for seg in classified_segments) / len(classified_segments)
            if avg_confidence < 0.6:
                recommendations.append("Low average classification confidence - consider manual content review")
        
        # Graph integration recommendations
        if graph_integration and len(graph_integration.content_mappings) == 0:
            recommendations.append("No content mapped to existing concepts - consider expanding knowledge graph")
        
        return recommendations
    
    def _generate_batch_summary(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """Generate summary statistics for batch processing"""
        successful_results = [r for r in results if r.success]
        
        summary = {
            'success_rate': len(successful_results) / len(results) if results else 0,
            'total_processing_time': sum(r.processing_time_seconds for r in results),
            'average_processing_time': sum(r.processing_time_seconds for r in results) / len(results) if results else 0,
            'content_statistics': {
                'total_equations': 0,
                'total_diagrams': 0,
                'total_concepts': 0,
                'difficulty_distribution': {'beginner': 0, 'intermediate': 0, 'advanced': 0}
            },
            'common_errors': [],
            'performance_metrics': {}
        }
        
        # Aggregate content statistics
        for result in successful_results:
            if result.processed_document:
                doc = result.processed_document
                summary['content_statistics']['total_equations'] += len(doc.content.equations)
                summary['content_statistics']['total_diagrams'] += len(doc.content.diagrams)
                summary['content_statistics']['total_concepts'] += len(doc.physics_concepts)
                
                difficulty = doc.difficulty_level
                if difficulty in summary['content_statistics']['difficulty_distribution']:
                    summary['content_statistics']['difficulty_distribution'][difficulty] += 1
        
        # Collect common errors
        failed_results = [r for r in results if not r.success]
        error_types = {}
        for result in failed_results:
            if result.error_message:
                error_key = result.error_message.split(':')[0]  # Get error type
                error_types[error_key] = error_types.get(error_key, 0) + 1
        
        summary['common_errors'] = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return summary
    
    def _save_graph_integration_result(self, integration_result: GraphIntegrationResult, output_path: str):
        """Save graph integration result to JSON"""
        result_dict = {
            'document_hash': integration_result.document_hash,
            'document_node_id': integration_result.document_node_id,
            'content_mappings': [
                {
                    'content_id': mapping.content_id,
                    'content_type': mapping.content_type,
                    'graph_concept_name': mapping.graph_concept_name,
                    'confidence_score': mapping.confidence_score,
                    'mapping_type': mapping.mapping_type,
                    'evidence': mapping.evidence
                }
                for mapping in integration_result.content_mappings
            ],
            'new_nodes_created': integration_result.new_nodes_created,
            'new_relationships_created': integration_result.new_relationships_created,
            'integration_metadata': integration_result.integration_metadata
        }
        
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
    
    def _save_final_result(self, result: ProcessingResult):
        """Save final processing result"""
        if not result.success:
            return
        
        output_path = os.path.join(
            self.config.output_directory,
            "processed_documents",
            f"final_result_{result.processed_document.document_hash[:8]}.json"
        )
        
        # Create a serializable version of the result
        result_dict = {
            'success': result.success,
            'file_path': result.file_path,
            'processing_time_seconds': result.processing_time_seconds,
            'warnings': result.warnings,
            'intermediate_files': result.intermediate_files,
            'document_summary': self.multimodal_processor.get_processing_summary(result.processed_document) if result.processed_document else None,
            'classification_summary': self.content_classifier.get_classification_summary(result.classified_segments) if result.classified_segments else None,
            'graph_integration_summary': {
                'document_node_id': result.graph_integration.document_node_id,
                'content_mappings_count': len(result.graph_integration.content_mappings),
                'new_nodes_count': len(result.graph_integration.new_nodes_created),
                'new_relationships_count': len(result.graph_integration.new_relationships_created)
            } if result.graph_integration else None
        }
        
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
    
    def _save_batch_results(self, batch_result: BatchProcessingResult):
        """Save batch processing results"""
        output_path = os.path.join(
            self.config.output_directory,
            f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        # Create serializable batch result
        batch_dict = {
            'total_files': batch_result.total_files,
            'successful_files': batch_result.successful_files,
            'failed_files': batch_result.failed_files,
            'total_processing_time': batch_result.total_processing_time,
            'batch_summary': batch_result.batch_summary,
            'file_results': [
                {
                    'file_path': result.file_path,
                    'success': result.success,
                    'processing_time': result.processing_time_seconds,
                    'error_message': result.error_message,
                    'warnings_count': len(result.warnings) if result.warnings else 0
                }
                for result in batch_result.processing_results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(batch_dict, f, indent=2)
        
        logger.info(f"Batch results saved to: {output_path}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and statistics"""
        return {
            'pipeline_config': asdict(self.config),
            'output_directory': self.config.output_directory,
            'components_enabled': {
                'multimodal_processing': True,
                'graph_integration': self.config.enable_graph_integration,
                'content_classification': self.config.enable_content_classification,
                'diagram_analysis': self.config.enable_diagram_analysis,
                'latex_processing': self.config.enable_latex_processing
            },
            'directories_created': os.path.exists(self.config.output_directory)
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.graph_integrator:
            self.graph_integrator.close()
        
        logger.info("Pipeline cleanup completed")

# Example usage and configuration
if __name__ == "__main__":
    # Example configuration
    config = PipelineConfig(
        output_directory="./physics_processing_output",
        enable_graph_integration=True,
        enable_content_classification=True,
        enable_diagram_analysis=True,
        enable_latex_processing=True,
        validate_results=True,
        save_intermediate_results=True,
        max_file_size_mb=50
    )
    
    pipeline = DocumentProcessingPipeline(config)
    
    try:
        status = pipeline.get_pipeline_status()
        print("Document Processing Pipeline initialized")
        print("Status:", json.dumps(status, indent=2))
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
    finally:
        pipeline.cleanup()