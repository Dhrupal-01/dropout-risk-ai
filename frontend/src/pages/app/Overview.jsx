import React from 'react';
import KpiCards from './overview/KpiCards';
import TierSplit from './overview/TierSplit';
import InterventionStatus from './overview/InterventionStatus';
import RiskHistogram from './overview/RiskHistogram';
import TopDrivers from './overview/TopDrivers';
import TierByDepartment from './overview/TierByDepartment';
import PriorityTable from './overview/PriorityTable';

// Cohort overview. Reads only /stats/* and /mentors/queue; every panel has loading, error and empty states.
const Overview = () => (
  <div className="px-4 sm:px-6 py-8 max-w-[1400px]">
    <h1 className="font-display font-medium text-32 tracking-display text-graphite">Overview</h1>
    <p className="mt-2 max-w-measure text-15 text-slate">
      The cohort at a glance. Risk levels are model estimates that help mentors decide whom to contact
      first; they are not decisions about any student.
    </p>

    <div className="mt-6">
      <KpiCards />
    </div>

    {/* items-start: a short panel keeps its own height instead of stretching to its neighbour's */}
    <div className="mt-4 grid grid-cols-1 items-start gap-4 lg:grid-cols-2">
      <TierSplit />
      <InterventionStatus />
      <RiskHistogram />
      <TierByDepartment />
      <TopDrivers className="lg:col-span-2" />
      <PriorityTable className="lg:col-span-2" />
    </div>
  </div>
);

export default Overview;
