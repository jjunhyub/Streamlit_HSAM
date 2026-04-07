import crypto from 'node:crypto';

const [, , ...pairs] = process.argv;

if (!pairs.length) {
  console.log('Usage: node scripts/hash-reviewer-passwords.mjs reviewerA=password1 reviewerB=password2');
  process.exit(0);
}

const output = {};
for (const pair of pairs) {
  const [reviewerId, ...rest] = pair.split('=');
  const password = rest.join('=');
  if (!reviewerId || !password) {
    throw new Error(`Invalid pair: ${pair}`);
  }

  const salt = crypto.randomBytes(16);
  const hash = crypto.scryptSync(password, salt, 64);
  output[reviewerId] = {
    passwordHash: `scrypt$${salt.toString('base64')}$${hash.toString('base64')}`,
  };
}

console.log(JSON.stringify(output, null, 2));
