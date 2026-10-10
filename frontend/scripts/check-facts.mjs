// Checks src/content/facts.json before a public deploy (npm run check:facts).
// Fails if any fact lacks publisher, source, year or url, or has a missing or duplicate id.
// Lists facts the owner has not verified yet (verified !== true); those do not fail the check.
// Usage: node scripts/check-facts.mjs [path/to/facts.json]
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const factsPath = resolve(process.argv[2] ?? resolve(here, '../src/content/facts.json'));
const { facts } = JSON.parse(readFileSync(factsPath, 'utf8'));

const problems = [];
const seenIds = new Set();
const isBlank = (value) => typeof value !== 'string' || value.trim() === '';

if (!Array.isArray(facts)) {
  problems.push('"facts" is not an array');
} else {
  facts.forEach((fact, index) => {
    const label = fact.id ? `"${fact.id}"` : `facts[${index}]`;
    if (isBlank(fact.id)) problems.push(`${label}: missing id`);
    else if (seenIds.has(fact.id)) problems.push(`${label}: duplicate id`);
    else seenIds.add(fact.id);

    for (const field of ['publisher', 'source', 'url']) {
      if (isBlank(fact[field])) problems.push(`${label}: missing ${field}`);
    }
    if (!Number.isInteger(fact.year)) problems.push(`${label}: missing or non-integer year`);
    if (!isBlank(fact.url) && !/^https?:\/\//.test(fact.url)) problems.push(`${label}: url is not http(s)`);
  });
}

const unverified = Array.isArray(facts) ? facts.filter((fact) => fact.verified !== true) : [];

console.log(`Checked ${Array.isArray(facts) ? facts.length : 0} facts in ${factsPath}`);
if (unverified.length) {
  console.log(`\n${unverified.length} unverified (open each URL and check the value before a public deploy):`);
  for (const fact of unverified) {
    console.log(`  - ${fact.id}: ${fact.url}`);
    if (fact.official_source_todo) console.log(`      todo: ${fact.official_source_todo}`);
  }
}

if (problems.length) {
  console.error(`\n${problems.length} problem(s):`);
  for (const problem of problems) console.error(`  - ${problem}`);
  process.exit(1);
}
console.log('\nAll facts have publisher, source, year and url.');
