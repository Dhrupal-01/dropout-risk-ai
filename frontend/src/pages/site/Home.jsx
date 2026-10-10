import React from 'react';
import { Link } from 'react-router-dom';

// Landing page. Build step 1 holds only the hero text; sections a–k arrive in steps 3–5.
const Home = () => (
  <section className="max-w-page mx-auto px-4 sm:px-6 py-16 md:py-28">
    <h1 className="font-display font-medium text-40 md:text-60 tracking-display text-graphite max-w-[18ch]">
      Most students don't drop out on one day.
    </h1>
    <p className="mt-6 max-w-measure text-17 md:text-20 text-graphite">
      They drift. A missed week, a fee left unpaid, an assignment that never comes in. The signs are
      already in your college's records. DropoutGuard brings them together, tells a mentor who may need
      support and why, while there is still time to help.
    </p>
    <div className="mt-8">
      <Link to="/app/overview" className="btn btn-primary">
        Open dashboard
      </Link>
    </div>
  </section>
);

export default Home;
