import React from 'react';

// Chart tooltip panel in the page tokens.
const TooltipBox = ({ title, children }) => (
  <div className="rounded-control border border-rule bg-paper px-3 py-2 text-13 text-graphite tabular-nums">
    {title && <p className="font-medium">{title}</p>}
    <div className="text-slate">{children}</div>
  </div>
);

export default TooltipBox;
