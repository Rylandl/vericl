//! Minimal seeded PRNG (SplitMix64) for reproducible differential-test inputs.
//! No external dependency so input generation stays stable across builds.

#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform f32 in [lo, hi).
    pub fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
        lo + (hi - lo) * unit
    }

    pub fn fill_f32(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_f32_range(lo, hi)).collect()
    }

    /// Uniform f64 in [lo, hi). Uses the top 53 bits of a fresh `u64` (f64's
    /// mantissa width) divided by 2^53, so the unit draw covers every
    /// representable dyadic in [0, 1) at full f64 precision — the f64 analog
    /// of `next_f32_range`'s 24-bit path (`>> 8`, `/ 2^24`).
    pub fn next_f64_range(&mut self, lo: f64, hi: f64) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + (hi - lo) * unit
    }

    pub fn fill_f64(&mut self, n: usize, lo: f64, hi: f64) -> Vec<f64> {
        (0..n).map(|_| self.next_f64_range(lo, hi)).collect()
    }

    pub fn fill_u32(&mut self, n: usize) -> Vec<u32> {
        (0..n).map(|_| self.next_u32()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn f32_in_range() {
        let mut r = SplitMix64::new(7);
        for _ in 0..1000 {
            let v = r.next_f32_range(-2.0, 3.0);
            assert!((-2.0..3.0).contains(&v));
        }
    }

    #[test]
    fn f64_in_range() {
        let mut r = SplitMix64::new(7);
        for _ in 0..1000 {
            let v = r.next_f64_range(-2.0, 3.0);
            assert!((-2.0..3.0).contains(&v));
        }
    }

    /// The 53-bit unit path yields values genuinely finer than f32 could
    /// represent (proving f64 generation is not silently going through f32):
    /// at least one draw in a wide sample differs from its own f32 round-trip.
    #[test]
    fn f64_uses_full_precision() {
        let mut r = SplitMix64::new(0xF64);
        let any_finer = (0..2000).any(|_| {
            let v = r.next_f64_range(0.0, 1.0);
            v != (v as f32) as f64
        });
        assert!(any_finer, "f64 draws should exceed f32 precision");
    }
}
