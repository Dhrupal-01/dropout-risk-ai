import React from 'react';
import { getSources } from '../content';

const Sources = () => (
  <section id="sources" aria-labelledby="sources-heading" className="border-t border-rule ruled-paper">
    <div className="max-w-page mx-auto px-4 sm:px-6 py-16">
      <h2 id="sources-heading" className="font-display font-medium text-32 leading-[4rem] tracking-display text-graphite">
        Sources
      </h2>
      {/* 32px line height keeps every line of text between two rules */}
      <ol className="max-w-measure text-15 leading-8 text-graphite list-decimal pl-6">
        {getSources().map((s) => (
          <li key={s.url} id={`source-${s.number}`} className="scroll-mt-24 pl-1">
            <span>{s.publisher}. </span>
            <a href={s.url} className="text-ink underline underline-offset-4 hover:no-underline break-words">
              {s.source}
            </a>
            <span className="text-slate"> ({s.year})</span>
          </li>
        ))}
      </ol>
    </div>
  </section>
);

export default Sources;
