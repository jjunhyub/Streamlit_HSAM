// @ts-nocheck

const REVIEWER_USERS = {
  admin: { password: '0000' },
  user1: { password: '5936' },
  user2: { password: '8226' },
  user3: { password: '5990' },
  user4: { password: '5823' },
  user5: { password: '4216' },
  user6: { password: '4520' },
  user7: { password: '3214' },
  user8: { password: '9291' },
  user9: { password: '9787' },
  user10: { password: '9348' },
  user11: { password: '1575' },
  user12: { password: '9336' },
  user13: { password: '1089' },
  user14: { password: '8303' },
  user15: { password: '2775' },
  user16: { password: '5267' },
  user17: { password: '6774' },
  user18: { password: '2621' },
  user19: { password: '8582' },
  user20: { password: '0842' },
};

export function getReviewerUsers() {
  return REVIEWER_USERS;
}

export function verifyReviewerPassword(reviewerId: string, password: string) {
  const entry = REVIEWER_USERS?.[reviewerId];
  if (!entry) return false;
  return entry.password === password;
}