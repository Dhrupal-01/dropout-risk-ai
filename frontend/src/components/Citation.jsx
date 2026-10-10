import React from 'react';
import { getFact, getSourceNumber } from '../content';

// In-text citation: a superscript number linking to the fact's entry in the Sources section.
// In development, facts the owner has not verified get a dotted underline and an "unverified" tooltip.
const Citation = ({ factId }) => {
  const fact = getFact(factId);
  if (!fact) return null;

  const number = getSourceNumber(factId);
  const flagUnverified = import.meta.env.DEV && fact.verified !== true;

  return (
    <sup className="ml-0.5 text-13 leading-none">
      <a
        href={`#source-${number}`}
        aria-label={`Source ${number}: ${fact.publisher}, ${fact.source}`}
        title={flagUnverified ? 'unverified' : undefined}
        className={`text-ink hover:underline ${flagUnverified ? 'border-b border-dotted border-current' : ''}`}
      >
        {number}
      </a>
    </sup>
  );
};

export default Citation;
