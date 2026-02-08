/**
 * Utility functions for formatting and cleaning up multi-agent response messages
 */

export interface FormattedMessage {
  content: string;
  agentUsed?: string;
  isFormatted: boolean;
}

/**
 * Cleans and formats a raw multi-agent response
 * @param rawResponse - The raw response from the multi-agent system
 * @param agentUsed - The agent that was primarily used (optional)
 * @returns Formatted message object
 */
export function formatMultiAgentResponse(rawResponse: string, agentUsed?: string): FormattedMessage {
  let cleanedResponse = rawResponse;

  // Remove any agent prefix patterns like "Agent: Policy & Procedures Specialist"
  cleanedResponse = cleanedResponse.replace(/^Agent:\s*[^\n]+\n*/gm, '');

  // Clean up markdown formatting for better readability
  cleanedResponse = formatMarkdownForDisplay(cleanedResponse);

  // Remove excessive whitespace and normalize line breaks
  cleanedResponse = cleanedResponse.replace(/\n{3,}/g, '\n\n');
  cleanedResponse = cleanedResponse.trim();

  // Extract agent information from the response if not provided
  let detectedAgent = agentUsed;
  const agentMatch = rawResponse.match(/Agent:\s*([^\n]+)/);
  if (agentMatch && !detectedAgent) {
    detectedAgent = agentMatch[1].trim();
  }

  return {
    content: cleanedResponse,
    agentUsed: detectedAgent,
    isFormatted: true
  };
}

/**
 * Formats markdown text for better display in chat interface
 * @param text - Text with markdown formatting
 * @returns Formatted text suitable for display
 */
function formatMarkdownForDisplay(text: string): string {
  let formatted = text;

  // Convert markdown headers to bold text
  formatted = formatted.replace(/^### (.+)$/gm, '**$1**');
  formatted = formatted.replace(/^## (.+)$/gm, '**$1**');
  formatted = formatted.replace(/^# (.+)$/gm, '**$1**');

  // Convert asterisk lists to bullet points
  formatted = formatted.replace(/^\* (.+)$/gm, '• $1');

  // Convert numbered lists to proper format
  formatted = formatted.replace(/^(\d+)\. (.+)$/gm, '$1. $2');

  // Convert **bold** to simple bold formatting (keeping ** for now, could be replaced with HTML later)
  // formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '$1'); // Uncomment if you want to remove bold markers

  // Clean up excessive formatting
  formatted = formatted.replace(/\*{3,}/g, '**'); // Replace *** with **

  return formatted;
}

/**
 * Extracts key information from a response for summary purposes
 * @param response - The formatted response
 * @returns Summary information
 */
export function extractResponseSummary(response: string): {
  hasNumbers: boolean;
  hasList: boolean;
  hasPolicy: boolean;
  hasEmployeeInfo: boolean;
} {
  return {
    hasNumbers: /\d+/.test(response),
    hasList: /^[•\-*\d+\.]\s/m.test(response),
    hasPolicy: /policy|procedure|guideline/i.test(response),
    hasEmployeeInfo: /employee|department|project/i.test(response)
  };
}

/**
 * Formats responses specifically for different query types
 * @param response - Raw response
 * @param queryType - Type of query ('employee', 'policy', 'organization', 'general')
 * @returns Formatted response
 */
export function formatResponseByType(response: string, queryType: string): string {
  let formatted = response;

  switch (queryType.toLowerCase()) {
    case 'employee':
    case 'department':
      // Format employee/department responses
      formatted = formatted.replace(/Employee\s+([^(]+)\s*\(ID:\s*([^)]+)\)/g, 
        '👤 **$1** (ID: $2)');
      break;

    case 'policy':
      // Format policy responses
      formatted = formatted.replace(/Policy:\s*([^\-]+)\s*-/g, 
        '📋 **Policy:** $1 -');
      break;

    case 'project':
      // Format project responses
      formatted = formatted.replace(/Project\s+([^:]+):/g, 
        '📊 **Project $1:**');
      break;

    default:
      // General formatting - no specific changes
      break;
  }

  return formatted;
}

/**
 * Validates if a response looks like debugging output
 * @param response - Response to check
 * @returns True if response appears to be debugging output
 */
export function isDebuggingResponse(response: string): boolean {
  const debugPatterns = [
    /Tool\s+execution/i,
    /JSONSearchTool/i,
    /PDFSearchTool/i,
    /CSVSearchTool/i,
    /CrewAI/i,
    /Agent\s+response:/i,
    /\[DEBUG\]/i,
    /Error\s+in\s+tool/i
  ];

  return debugPatterns.some(pattern => pattern.test(response));
}

/**
 * Provides a fallback message when debugging response is detected
 * @param queryType - Type of query that was asked
 * @returns User-friendly fallback message
 */
export function getDebugFallbackMessage(queryType: string): string {
  const fallbacks: Record<string, string> = {
    employee: "I'm currently processing your employee-related question. Our system is analyzing the employee database to provide you with accurate information.",
    policy: "I'm reviewing the company policies to answer your question. Please bear with me as I search through our policy documents.",
    organization: "I'm gathering organizational information to provide you with comprehensive details about the company structure.",
    default: "I'm processing your question and searching through our company data. Please wait while I compile the most relevant information for you."
  };

  return fallbacks[queryType.toLowerCase()] || fallbacks.default;
}
