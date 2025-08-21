/**
 * Utility functions for data export functionality
 */

import { saveAs } from 'file-saver';
import * as XLSX from 'xlsx';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import { StudentInsights, ClassOverview, ExportFormat } from '../types/api';

/**
 * Export student data to various formats
 */
export const exportStudentData = async (
  data: StudentInsights,
  studentId: string,
  format: ExportFormat = 'json'
): Promise<void> => {
  const timestamp = new Date().toISOString().split('T')[0];
  const filename = `student_${studentId}_${timestamp}`;

  switch (format) {
    case 'json':
      exportAsJSON(data, `${filename}.json`);
      break;
    case 'csv':
      exportStudentAsCSV(data, `${filename}.csv`);
      break;
    case 'excel':
      exportStudentAsExcel(data, `${filename}.xlsx`);
      break;
  }
};

/**
 * Export class data to various formats
 */
export const exportClassData = async (
  data: ClassOverview,
  classIds: string[],
  format: ExportFormat = 'json'
): Promise<void> => {
  const timestamp = new Date().toISOString().split('T')[0];
  const filename = `class_overview_${classIds.join('_')}_${timestamp}`;

  switch (format) {
    case 'json':
      exportAsJSON(data, `${filename}.json`);
      break;
    case 'csv':
      exportClassAsCSV(data, `${filename}.csv`);
      break;
    case 'excel':
      exportClassAsExcel(data, `${filename}.xlsx`);
      break;
  }
};

/**
 * Export data as JSON file
 */
export const exportAsJSON = (data: any, filename: string): void => {
  const jsonString = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonString], { type: 'application/json' });
  saveAs(blob, filename);
};

/**
 * Export student data as CSV
 */
export const exportStudentAsCSV = (data: StudentInsights, filename: string): void => {
  const headers = [
    'Student ID',
    'Overall Score',
    'Learning Velocity',
    'Engagement Score',
    'Concepts Mastered',
    'Total Concepts',
    'Generated At'
  ];

  const row = [
    data.user_id,
    data.progress_tracking.overall_score.toString(),
    data.progress_tracking.learning_velocity.toString(),
    data.progress_tracking.engagement_score.toString(),
    data.progress_tracking.concepts_mastered.toString(),
    data.progress_tracking.total_concepts.toString(),
    data.generated_at
  ];

  // Add concept mastery data
  const conceptHeaders = ['Concept', 'Mastery Score', 'Confidence Min', 'Confidence Max'];
  const conceptRows = data.concept_mastery.map(concept => [
    concept.concept,
    concept.mastery_score.toString(),
    concept.confidence[0].toString(),
    concept.confidence[1].toString()
  ]);

  let csvContent = headers.join(',') + '\n';
  csvContent += row.join(',') + '\n\n';
  csvContent += conceptHeaders.join(',') + '\n';
  csvContent += conceptRows.map(row => row.join(',')).join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8' });
  saveAs(blob, filename);
};

/**
 * Export class data as CSV
 */
export const exportClassAsCSV = (data: ClassOverview, filename: string): void => {
  const headers = [
    'Total Students',
    'Total Interactions',
    'Average Response Time',
    'Class Success Rate',
    'Most Used Agent',
    'Generated At'
  ];

  const row = [
    data.class_statistics.total_students.toString(),
    data.class_statistics.total_interactions.toString(),
    data.class_statistics.avg_response_time.toString(),
    data.class_statistics.class_success_rate.toString(),
    data.class_statistics.most_used_agent,
    data.generated_at
  ];

  // Add top performers data
  const performerHeaders = ['User ID', 'Success Rate', 'Interaction Count'];
  const performerRows = data.top_performers.map(performer => [
    performer.user_id,
    performer.success_rate.toString(),
    performer.interaction_count.toString()
  ]);

  let csvContent = headers.join(',') + '\n';
  csvContent += row.join(',') + '\n\n';
  csvContent += performerHeaders.join(',') + '\n';
  csvContent += performerRows.map(row => row.join(',')).join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8' });
  saveAs(blob, filename);
};

/**
 * Export student data as Excel file
 */
export const exportStudentAsExcel = (data: StudentInsights, filename: string): void => {
  const workbook = XLSX.utils.book_new();

  // Summary sheet
  const summaryData = [
    ['Student ID', data.user_id],
    ['Overall Score', data.progress_tracking.overall_score],
    ['Learning Velocity', data.progress_tracking.learning_velocity],
    ['Engagement Score', data.progress_tracking.engagement_score],
    ['Concepts Mastered', data.progress_tracking.concepts_mastered],
    ['Total Concepts', data.progress_tracking.total_concepts],
    ['Generated At', data.generated_at],
  ];

  const summarySheet = XLSX.utils.aoa_to_sheet(summaryData);
  XLSX.utils.book_append_sheet(workbook, summarySheet, 'Summary');

  // Concept mastery sheet
  const conceptData = [
    ['Concept', 'Mastery Score', 'Confidence Min', 'Confidence Max'],
    ...data.concept_mastery.map(concept => [
      concept.concept,
      concept.mastery_score,
      concept.confidence[0],
      concept.confidence[1]
    ])
  ];

  const conceptSheet = XLSX.utils.aoa_to_sheet(conceptData);
  XLSX.utils.book_append_sheet(workbook, conceptSheet, 'Concept Mastery');

  // Predictions sheet (if available)
  if (data.predictions) {
    const predictionsData = [
      ['Metric', 'Predicted Value', 'Confidence', 'Factors'],
      [
        'Success Rate',
        data.predictions.success_rate.predicted_value,
        data.predictions.success_rate.confidence,
        data.predictions.success_rate.factors.join('; ')
      ]
    ];

    if (data.predictions.learning_velocity) {
      predictionsData.push([
        'Learning Velocity',
        data.predictions.learning_velocity.predicted_value,
        data.predictions.learning_velocity.confidence,
        data.predictions.learning_velocity.factors.join('; ')
      ]);
    }

    if (data.predictions.engagement_score) {
      predictionsData.push([
        'Engagement Score',
        data.predictions.engagement_score.predicted_value,
        data.predictions.engagement_score.confidence,
        data.predictions.engagement_score.factors.join('; ')
      ]);
    }

    const predictionsSheet = XLSX.utils.aoa_to_sheet(predictionsData);
    XLSX.utils.book_append_sheet(workbook, predictionsSheet, 'Predictions');
  }

  XLSX.writeFile(workbook, filename);
};

/**
 * Export class data as Excel file
 */
export const exportClassAsExcel = (data: ClassOverview, filename: string): void => {
  const workbook = XLSX.utils.book_new();

  // Summary sheet
  const summaryData = [
    ['Total Students', data.class_statistics.total_students],
    ['Total Interactions', data.class_statistics.total_interactions],
    ['Average Response Time', data.class_statistics.avg_response_time],
    ['Class Success Rate', data.class_statistics.class_success_rate],
    ['Most Used Agent', data.class_statistics.most_used_agent],
    ['Generated At', data.generated_at],
  ];

  const summarySheet = XLSX.utils.aoa_to_sheet(summaryData);
  XLSX.utils.book_append_sheet(workbook, summarySheet, 'Summary');

  // Performance distribution sheet
  const distributionData = [
    ['Metric', 'Value'],
    ['25th Percentile', data.performance_distribution.percentiles['25th']],
    ['50th Percentile', data.performance_distribution.percentiles['50th']],
    ['75th Percentile', data.performance_distribution.percentiles['75th']],
    ['90th Percentile', data.performance_distribution.percentiles['90th']],
    ['Student Count', data.performance_distribution.student_count],
  ];

  const distributionSheet = XLSX.utils.aoa_to_sheet(distributionData);
  XLSX.utils.book_append_sheet(workbook, distributionSheet, 'Distribution');

  // Top performers sheet
  const performersData = [
    ['User ID', 'Success Rate', 'Interaction Count'],
    ...data.top_performers.map(performer => [
      performer.user_id,
      performer.success_rate,
      performer.interaction_count
    ])
  ];

  const performersSheet = XLSX.utils.aoa_to_sheet(performersData);
  XLSX.utils.book_append_sheet(workbook, performersSheet, 'Top Performers');

  XLSX.writeFile(workbook, filename);
};

/**
 * Generate PDF report for students
 */
export const generatePDFReport = async (dataType: 'students' | 'classes' | 'system'): Promise<void> => {
  const pdf = new jsPDF();
  
  // Add title
  pdf.setFontSize(20);
  pdf.text(`Physics Assistant ${dataType.charAt(0).toUpperCase() + dataType.slice(1)} Report`, 20, 30);
  
  // Add timestamp
  pdf.setFontSize(12);
  pdf.text(`Generated: ${new Date().toLocaleDateString()}`, 20, 50);
  
  // Add content based on data type
  switch (dataType) {
    case 'students':
      pdf.text('Student Analytics Summary', 20, 70);
      pdf.text('• Overall performance metrics', 25, 85);
      pdf.text('• Concept mastery breakdown', 25, 95);
      pdf.text('• Learning progression analysis', 25, 105);
      pdf.text('• Engagement patterns', 25, 115);
      break;
      
    case 'classes':
      pdf.text('Class Overview Summary', 20, 70);
      pdf.text('• Class performance distribution', 25, 85);
      pdf.text('• Top performer analysis', 25, 95);
      pdf.text('• Interaction statistics', 25, 105);
      pdf.text('• Success rate trends', 25, 115);
      break;
      
    case 'system':
      pdf.text('System Metrics Summary', 20, 70);
      pdf.text('• Performance indicators', 25, 85);
      pdf.text('• Cache statistics', 25, 95);
      pdf.text('• System health status', 25, 105);
      pdf.text('• Resource utilization', 25, 115);
      break;
  }
  
  // Add disclaimer
  pdf.setFontSize(10);
  pdf.text('This report was automatically generated by the Physics Assistant Dashboard.', 20, 250);
  
  // Save the PDF
  const filename = `${dataType}_report_${new Date().toISOString().split('T')[0]}.pdf`;
  pdf.save(filename);
};

/**
 * Generate Excel report with multiple sheets
 */
export const generateExcelReport = async (dataType: 'students' | 'classes' | 'system'): Promise<void> => {
  const workbook = XLSX.utils.book_new();
  
  // Create sample data based on type
  const sampleData = generateSampleReportData(dataType);
  
  Object.entries(sampleData).forEach(([sheetName, data]) => {
    const sheet = XLSX.utils.aoa_to_sheet(data);
    XLSX.utils.book_append_sheet(workbook, sheet, sheetName);
  });
  
  const filename = `${dataType}_report_${new Date().toISOString().split('T')[0]}.xlsx`;
  XLSX.writeFile(workbook, filename);
};

/**
 * Generate sample report data
 */
const generateSampleReportData = (dataType: string): Record<string, any[][]> => {
  const timestamp = new Date().toISOString();
  
  switch (dataType) {
    case 'students':
      return {
        'Overview': [
          ['Metric', 'Value'],
          ['Total Students', '150'],
          ['Active Students', '142'],
          ['Average Progress', '78.5%'],
          ['Generated At', timestamp],
        ],
        'Performance': [
          ['Student ID', 'Overall Score', 'Engagement', 'Concepts Mastered'],
          ['student_001', '85', '92', '12'],
          ['student_002', '78', '88', '10'],
          ['student_003', '92', '95', '14'],
        ],
      };
      
    case 'classes':
      return {
        'Overview': [
          ['Metric', 'Value'],
          ['Total Classes', '5'],
          ['Total Students', '150'],
          ['Average Performance', '82.3%'],
          ['Generated At', timestamp],
        ],
        'Class Performance': [
          ['Class ID', 'Students', 'Avg Score', 'Completion Rate'],
          ['Physics 101A', '30', '85', '95%'],
          ['Physics 101B', '28', '78', '89%'],
          ['Physics 102A', '25', '92', '96%'],
        ],
      };
      
    default:
      return {
        'System Health': [
          ['Component', 'Status', 'Uptime'],
          ['API Server', 'Healthy', '99.9%'],
          ['Database', 'Healthy', '99.8%'],
          ['Cache', 'Healthy', '99.7%'],
        ],
      };
  }
};

/**
 * Export chart as image
 */
export const exportChartAsImage = async (
  chartElementId: string,
  filename: string,
  format: 'png' | 'jpeg' = 'png'
): Promise<void> => {
  const element = document.getElementById(chartElementId);
  if (!element) {
    throw new Error(`Element with ID ${chartElementId} not found`);
  }

  const canvas = await html2canvas(element);
  canvas.toBlob((blob) => {
    if (blob) {
      saveAs(blob, `${filename}.${format}`);
    }
  }, `image/${format}`);
};