import React from 'react';
import { Link } from 'react-router-dom';
import PreviewCards from './PreviewCards';

// "See how it works" joins the buttons when the #how-it-works section exists (phase 4).
const Hero = () => (
  <section className="max-w-page mx-auto px-4 sm:px-6 py-16 md:py-24 grid gap-12 lg:grid-cols-12 lg:items-center">
    <div className="lg:col-span-6">
      <h1 className="font-display font-medium text-40 md:text-60 tracking-display text-graphite max-w-[18ch]">
        Most students don't drop out on one day.
      </h1>
      <p className="mt-6 max-w-measure text-17 md:text-20 text-graphite">
        They drift. A missed week, a fee left unpaid, an assignment that never comes in. The signs are
        already in your college's records. DropoutGuard brings them together, tells a mentor who may need
        support and why, while there is still time to help.
      </p>
      <div className="mt-8 flex flex-wrap gap-3">
        <Link to="/app/overview" className="btn btn-primary">
          Open dashboard
        </Link>
      </div>
    </div>
    <div className="lg:col-span-6">
      <PreviewCards />
    </div>
  </section>
);

export default Hero;
