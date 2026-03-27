import katex from 'katex';
import { useMemo } from 'react';

interface MathBlockProps {
  math: string;
  block?: boolean;
  className?: string;
}

export function MathBlock({ math, block = true, className = '' }: MathBlockProps) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(math, {
        displayMode: block,
        throwOnError: false,
      });
    } catch (e) {
      console.error('KaTeX error:', e);
      return math;
    }
  }, [math, block]);

  return (
    <span
      className={`inline-block ${block ? 'my-4 text-xl md:text-2xl' : ''} ${className}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
