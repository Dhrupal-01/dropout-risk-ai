import React from 'react';
import Sources from '../../components/Sources';
import Hero from './landing/Hero';
import ProblemNumbers from './landing/ProblemNumbers';
import ProblemCharts from './landing/ProblemCharts';
import Register from './landing/Register';
import Features from './landing/Features';
import HowItWorks from './landing/HowItWorks';
import Evidence from './landing/Evidence';
import ResponsibleAI from './landing/ResponsibleAI';
import Faq from './landing/Faq';
import FinalCta from './landing/FinalCta';

// Landing page in Revision 2 order (a–k). Motion arrives in phase 5.
const Home = () => (
  <>
    <Hero />
    <ProblemNumbers />
    <ProblemCharts />
    <Register />
    <Features />
    <HowItWorks />
    <Evidence />
    <ResponsibleAI />
    <Faq />
    <FinalCta />
    <Sources />
  </>
);

export default Home;
