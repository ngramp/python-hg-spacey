final result
--- 
- party
- party relationship, with
  - entity as party (org, company, person)
  - topic as party
  - url and archive link as links

rules
---
- only one relation regardless of multiple mentions
- leave possible spark for future scaling

strat
--- 
- [x] get text from warc
- [x] preprocess text
- [x] get entities from text
- [x] get topics from text - yes but expensive, maybe do on demand

__
- [ ] form party relationships
- [ ] discard articles with no entities
- [ ] save party relationships with article clean text and ref back to warc
