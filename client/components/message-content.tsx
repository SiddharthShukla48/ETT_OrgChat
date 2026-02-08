"use client"

import type React from "react"

interface MessageContentProps {
  content: string
  className?: string
}

export function MessageContent({ content, className = "" }: MessageContentProps) {
  // Split content by double newlines to create paragraphs
  const paragraphs = content.split('\n\n').filter(p => p.trim());
  
  const formatText = (text: string): React.ReactNode => {
    // Handle bold text **text**
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    
    return parts.map((part, index) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return (
          <strong key={index} className="font-semibold text-foreground">
            {part.slice(2, -2)}
          </strong>
        );
      }
      return part;
    });
  };

  const formatParagraph = (paragraph: string, index: number): React.ReactNode => {
    const lines = paragraph.split('\n');
    
    return (
      <div key={index} className="mb-3 last:mb-0">
        {lines.map((line, lineIndex) => {
          // Handle bullet points
          if (line.match(/^[•\-*]\s/)) {
            return (
              <div key={lineIndex} className="ml-4 mb-1 flex items-start">
                <span className="text-blue-500 mr-2 mt-1 text-xs">•</span>
                <span className="flex-1">{formatText(line.replace(/^[•\-*]\s/, ''))}</span>
              </div>
            );
          }
          
          // Handle numbered lists
          if (line.match(/^\d+\.\s/)) {
            const [, number, text] = line.match(/^(\d+)\.\s(.+)$/) || [];
            return (
              <div key={lineIndex} className="ml-4 mb-1 flex items-start">
                <span className="text-blue-500 mr-2 min-w-[1.5rem] text-sm font-medium">{number}.</span>
                <span className="flex-1">{formatText(text || '')}</span>
              </div>
            );
          }
          
          // Handle emoji prefixed items (👤, 📋, 📊)
          if (line.match(/^[👤📋📊]/)) {
            return (
              <div key={lineIndex} className="mb-1 p-2 bg-blue-50 dark:bg-blue-950/20 rounded-md border-l-2 border-blue-200 dark:border-blue-800">
                {formatText(line)}
              </div>
            );
          }
          
          // Handle regular lines
          if (line.trim()) {
            return (
              <div key={lineIndex} className="mb-1">
                {formatText(line)}
              </div>
            );
          }
          
          return null;
        })}
      </div>
    );
  };

  return (
    <div className={`text-sm leading-relaxed ${className}`}>
      {paragraphs.map(formatParagraph)}
    </div>
  );
}
