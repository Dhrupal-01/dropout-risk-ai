import React from 'react';
import Sources from '../../components/Sources';
import Hero from './landing/Hero';
import ProblemNumbers from './landing/ProblemNumbers';
import ProblemCharts from './landing/ProblemCharts';

// Landing page, Revision 2 order. Built so far: a (hero), b (problem in numbers), c (problem charts)
// and the Sources section; d–j follow in phase 4, motion in phase 5.
const Home = () => (
  <>
    <Hero />
    <ProblemNumbers />
    <ProblemCharts />
    <Sources />
  </>
);

export default Home;
