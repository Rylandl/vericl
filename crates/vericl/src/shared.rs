//! Poison-initialized shared-memory model for the cooperative reference twin.
//!
//! A workgroup-cooperative kernel's `SharedMemory::<T>::new(N)` tile is
//! *uninitialized* on the GPU until a thread writes it (docs/design-shared-
//! memory.md §4.5). The phase-split reference twin (see `vericl-macros`'
//! cooperative twin derivation) models one such tile as a per-cube
//! [`SharedTile`]: every cell starts **poisoned**, and a read of a cell no
//! earlier segment wrote **panics loudly** — exactly the out-of-bounds-read
//! panic the bounds twin already relies on (README "First finding" /
//! `axpy_off_by_one`), so a shared-memory *definedness* bug surfaces as a
//! reference panic (a reported finding) instead of silently reading a zero
//! the GPU would read as garbage.
//!
//! Why this shape works with ordinary Rust indexing in a derived twin body:
//!
//! - A **read** — `let a = tile[i];`, `tile[i] + tile[j]`, any value-position
//!   use — resolves through [`Index`], which asserts the cell was written and
//!   panics on poison.
//! - A **write** — `tile[i] = v;` — resolves through [`IndexMut`], which marks
//!   the cell written and hands back a live `&mut T` to store into.
//!
//! Rust always prefers `Index` for a value-position read of a `Copy` element
//! and `IndexMut` only for a place expression (assignment target), so the twin
//! never has to distinguish the two: the read/write split falls out of the
//! language. A read-modify-write (`tile[i] += v`) would go through `IndexMut`
//! and thus *not* poison-check its read half — the cooperative twin subset
//! rejects compound assignment to a shared tile at macro time for exactly this
//! reason (see `vericl-macros`' cooperative recognizer), so it cannot arise
//! here.

use std::ops::{Index, IndexMut};

/// A single per-cube shared-memory tile for the reference twin, with
/// poison (read-before-write) detection. Backed by a dense `Vec<T>` plus a
/// parallel written-mask; a read of an unwritten cell panics.
///
/// `T: Default + Clone` because the backing store needs concrete initial
/// values to hand out `&mut T` on a first write — the poison state is tracked
/// separately in `written`, not by the element value, so the default is never
/// observable through a (checked) read.
pub struct SharedTile<T> {
    data: Vec<T>,
    written: Vec<bool>,
}

impl<T: Default + Clone> SharedTile<T> {
    /// A fresh, fully-poisoned tile of `len` cells (one per cube, per §4.5).
    pub fn new_poison(len: usize) -> Self {
        Self { data: vec![T::default(); len], written: vec![false; len] }
    }

    /// Number of cells (the `SharedMemory::<T>::new(N)` length).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tile has zero cells (present so `len()` doesn't draw a
    /// clippy `len_without_is_empty`).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<T> Index<usize> for SharedTile<T> {
    type Output = T;

    fn index(&self, i: usize) -> &T {
        assert!(
            self.written[i],
            "vericl: shared-memory read of poison (uninitialized) cell {i} — the kernel reads \
             shared memory before any thread writes it (docs/design-shared-memory.md §4.5). On \
             the GPU this reads uninitialized memory (garbage); the twin surfaces it as a \
             reference panic rather than masking it with a zero."
        );
        &self.data[i]
    }
}

impl<T> IndexMut<usize> for SharedTile<T> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        self.written[i] = true;
        &mut self.data[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn written_cell_reads_back() {
        let mut t = SharedTile::<f32>::new_poison(4);
        t[2] = 3.5;
        assert_eq!(t[2], 3.5);
    }

    #[test]
    #[should_panic(expected = "poison")]
    fn poison_read_panics() {
        let t = SharedTile::<f32>::new_poison(4);
        let _ = t[1]; // never written -> loud panic, not a silent zero
    }

    #[test]
    fn write_then_read_neighbor_is_independent() {
        let mut t = SharedTile::<f32>::new_poison(4);
        t[0] = 1.0;
        t[1] = 2.0;
        assert_eq!(t[0] + t[1], 3.0);
    }

    #[test]
    #[should_panic(expected = "poison")]
    fn partial_write_still_poisons_unwritten_cells() {
        let mut t = SharedTile::<f32>::new_poison(4);
        t[0] = 9.0; // only cell 0 written
        let _ = t[3]; // reading an unwritten cell still panics
    }
}
