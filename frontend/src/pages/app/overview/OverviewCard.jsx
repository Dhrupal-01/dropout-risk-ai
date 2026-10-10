import React, { useId } from 'react';
import { AlertCircle, RotateCw } from 'lucide-react';

export const Skeleton = ({ className = '' }) => (
  <div role="status" aria-label="Loading" className={`motion-safe:animate-pulse rounded-control bg-ink-wash ${className}`} />
);

export const ErrorState = ({ error, onRetry }) => (
  <div className="flex flex-col items-start gap-2 rounded-control border border-rule p-3 text-15 text-graphite">
    <p className="flex items-start gap-2">
      <AlertCircle className="mt-0.5 w-4 h-4 shrink-0 text-slate" aria-hidden="true" />
      <span>
        Couldn't load this. {error?.message || 'Check the backend status in the top bar.'}
      </span>
    </p>
    {onRetry && (
      <button type="button" onClick={onRetry} className="btn btn-secondary py-1 text-13">
        <RotateCw className="w-3.5 h-3.5" aria-hidden="true" />
        Try again
      </button>
    )}
  </div>
);

export const EmptyState = ({ children }) => (
  <div className="rounded-control border border-dashed border-rule p-4 text-15 text-slate">{children}</div>
);

// One panel of the Overview. Every panel has the same four states: loading, error, empty, ready.
const OverviewCard = ({
  title,
  description,
  status,
  error,
  onRetry,
  isEmpty = false,
  emptyMessage,
  loadingClassName = 'h-56',
  headerExtra,
  className = '',
  children,
}) => {
  const headingId = useId();
  let body;
  if (status === 'pending') body = <Skeleton className={loadingClassName} />;
  else if (status === 'error') body = <ErrorState error={error} onRetry={onRetry} />;
  else if (isEmpty) body = <EmptyState>{emptyMessage}</EmptyState>;
  else body = children;

  return (
    <section aria-labelledby={headingId} className={`min-w-0 rounded-panel border border-rule bg-paper p-4 sm:p-5 ${className}`}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 id={headingId} className="text-17 font-semibold text-graphite">
            {title}
          </h2>
          {description && <p className="mt-0.5 text-13 text-slate max-w-measure">{description}</p>}
        </div>
        {headerExtra}
      </div>
      <div className="mt-4">{body}</div>
    </section>
  );
};

export default OverviewCard;
