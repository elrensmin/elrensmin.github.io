---
title: geohash notes
date: 2024-11-20
description: "binary bisect of the globe, interleaved bits, base32: how geohash turns a lat/lon into a proximity-searchable string"
tags: [algorithms, gis, study-notes]
---

a **geohash** identifies a rectangular area around a fixed point, and the *length* of the hash sets the precision. long shared prefix ⇒ spatially close; chop characters off the end ⇒ gradually coarser cell. that asymmetry is the whole design: prefix similarity is a *sufficient* signal for proximity, never a guarantee - two points on opposite sides of a cell boundary (or the 180° meridian, or a pole) can be 10 m apart with wildly different codes.

my source note collected an unsourced "who uses geohash" list (Yelp, Google Maps, Uber, Airbnb) - dropped. the mechanics below are all that survived the port, plus a re-derived worked example.

## POI search via geohash

the proximity trick behind every "nearby" tab:

- get the user's lat/lon
- compute their geohash
- choose the precision: longer prefix = tighter cell = fewer, closer results; truncate the other way for wider search
- match prefix against the indexed POIs (all indexed by geohash prefix)
- tune the length until the candidate count is sane (you always re-filter by true distance afterwards - prefix match only proves *which cell*, not distance)

## the algorithm

bisect the planet. maintain a bounding box for latitude (-90, 90) and longitude (-180, 180). pick the target's midpoint; whichever half it falls in, record **1** and narrow the box to that half, else record **0**. alternate between longitude and latitude with every bit (longitude first), reading binary digits MSB-first (16, 8, 4, 2, 1). every **5 bits** → one base32 character → one grid cell. shorter string = fewer bits = bigger cells = lower precision.

the base32 alphabet is `0123456789bcdefghjkmnpqrstuvwxyz` - digits 0-9 plus 22 lowercase letters, with **a, i, l, o excluded** to cut transcription errors (my vault wrote "excluding vowels" and got it wrong: i/l/o yes, but also a; and e/u are kept. the code snippet at the bottom of my vault note had the alphabet right and the prose wrong).

## the worked example, recomputed

my vault's example ("latitude 37.7749° N, longitude 122.4194° W → geohash `w15df6577r2tsllft4`") was wrong twice over: it treated west longitude as positive, and the interleaved bit string was fabricated. here is the correct walkthrough, computed with the algorithm above.

location: latitude 37.7749, longitude **-122.4194** (west is negative!).

bisect longitude/latitude alternately, 40 bits (longitude gets the even bits):

- interleaved: `0100110110010001111011110100100100011110`
- take 5 bits at a time and look up base32:

| bits | value | char |
|---|---|---|
| 01001 | 9 | 9 |
| 10110 | 22 | q |
| 01000 | 8 | 8 |
| 11110 | 30 | y |
| 11110 | 30 | y |
| 10010 | 18 | k |
| 01000 | 8 | 8 |
| 11110 | 30 | y |

geohash: **`9q8yyk8y`** (the well-known san francisco prefix `9q8yy` falls out of the first 20 bits - a good sanity check).

## precision → cell size

computed cell dimensions (km, lon side scaled by cos(latitude), shown at 37.77°):

| chars | lon bits / lat bits | cell size |
|---|---|---|
| 1 | 2 / 3 | ~7900 × 2500 km |
| 2 | 5 / 5 | ~990 × 625 km |
| 3 | 7 / 8 | ~250 × 78 km |
| 4 | 10 / 10 | ~31 × 20 km |
| 5 | 12 / 13 | ~7.7 × 2.4 km |
| 6 | 15 / 15 | ~970 m × 610 m (≈0.6 km² - my vault's "0.61 square kilometers", kept) |
| 7 | 17 / 18 | ~240 × 76 m |
| 8 | 20 / 20 | ~30 × 19 m |

note the pattern: 5 bits per char alternate lon/lat starting with lon, so odd precisions give latitude one extra bit (bigger cells straddle the globe's east-west extent asymmetrically).

## implementation

my vault's python was broken in four ways at once (encode used 5 longitude bits *then* 5 latitude bits per character instead of alternating; `decode` was indented outside the class; `self.__base32` referenced a name-mangled attribute that didn't exist; the alphabet constant lived at module bottom in a way that wouldn't import). this is the corrected version, verified against the worked example above:

```python
class GeoHash:
    base32 = '0123456789bcdefghjkmnpqrstuvwxyz'

    def __init__(self, latitude, longitude, precision=12):
        self.latitude, self.longitude, self.precision = latitude, longitude, precision

    def encode(self):
        lat_lo, lat_hi, lon_lo, lon_hi = -90.0, 90.0, -180.0, 180.0
        bits = [16, 8, 4, 2, 1]
        chars, ch, bit, even = [], 0, 0, True
        while len(chars) < self.precision:
            mid = (lon_lo + lon_hi) / 2 if even else (lat_lo + lat_hi) / 2
            value = self.longitude if even else self.latitude
            if value >= mid:
                ch |= bits[bit]
                if even: lon_lo = mid
                else: lat_lo = mid
            else:
                if even: lon_hi = mid
                else: lat_hi = mid
            even = not even
            if bit < 4:
                bit += 1
            else:
                chars.append(GeoHash.base32[ch])
                ch, bit = 0, 0
        return ''.join(chars)

    @classmethod
    def decode(cls, geohash):
        lat_lo, lat_hi, lon_lo, lon_hi, even = -90.0, 90.0, -180.0, 180.0, True
        for c in geohash:
            cd = cls.base32.index(c)
            for j in range(5):
                mask = 1 << (4 - j)
                if even:
                    if cd & mask: lon_lo = (lon_lo + lon_hi) / 2
                    else: lon_hi = (lon_lo + lon_hi) / 2
                else:
                    if cd & mask: lat_lo = (lat_lo + lat_hi) / 2
                    else: lat_hi = (lat_lo + lat_hi) / 2
                even = not even
        return (lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2
```

## edge cases

- **antimeridian**: two points a meter apart on opposite sides of ±180° get hashes with *no shared prefix*.
- **poles**: latitude collapses near ±90°, longitude becomes meaningless - hashes for neighbors diverge.
- **bounded-box search**: computing the southwest/northeast corner hashes and scanning everything between retrieves *all* points along the z-order curve between those corners, which can be orders of magnitude too many (and still breaks at the meridian/poles). production systems keep a **filter list of the nearby cells' prefixes** (solr-style) instead of a numeric range scan.

## papers & sources

- geohash system explanation: <https://en.wikipedia.org/wiki/Geohash>
- a clean blog walkthrough i kept: <https://eugene-eeo.github.io/blog/geohashing.html>

## todos

- neighbor cells (geohash `n/s/e/w` neighbor calculation for a given hash) - the missing piece for doing real proximity searches without range scans.
- geohash vs. other space-filling curves (H3, S2) - when the hierarchical-prefix trick stops beating a proper tiling.
- cross-link a note when i write about z-order curves properly.