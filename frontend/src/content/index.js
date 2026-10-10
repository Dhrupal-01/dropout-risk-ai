import factsData from './facts.json';

// Every statistic on the site is read through here, never typed into JSX.
const factsById = new Map(factsData.facts.map((fact) => [fact.id, fact]));

// One numbered source per distinct URL, in the order the URL first appears in facts.json.
const sources = [];
const sourceNumberByUrl = new Map();
for (const fact of factsData.facts) {
  let source = sources[sourceNumberByUrl.get(fact.url) - 1];
  if (!source) {
    source = {
      number: sources.length + 1,
      url: fact.url,
      publisher: fact.publisher,
      source: fact.source,
      year: fact.year,
      factIds: [],
      verified: true,
    };
    sources.push(source);
    sourceNumberByUrl.set(fact.url, source.number);
  }
  source.factIds.push(fact.id);
  source.verified = source.verified && fact.verified === true;
}

// Unknown ids throw in development so a typo fails loudly; production returns undefined.
export const getFact = (id) => {
  const fact = factsById.get(id);
  if (!fact && import.meta.env.DEV) {
    throw new Error(`Unknown fact id "${id}". Add it to src/content/facts.json or fix the id.`);
  }
  return fact;
};

export const getSources = () => sources;

export const getSourceNumber = (factId) => {
  const fact = getFact(factId);
  return fact ? sourceNumberByUrl.get(fact.url) : undefined;
};
