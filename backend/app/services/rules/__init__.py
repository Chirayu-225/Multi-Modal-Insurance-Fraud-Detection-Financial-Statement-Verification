"""
Rule packs for the cross-verification engine.

  claims.py        the original 9-check motor-claim engine
                   (services/consistency.py), re-exported unchanged
  underwriting.py  the commercial-underwriting pack added by the merge
  scoring.py       noisy-OR combination shared by both
  registry.py      one entry point that picks the pack by case type
"""
