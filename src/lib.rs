//! Candle-Video: LTX-Video integration for Candle framework.
//!
//! This crate provides Rust implementations of video generation models,
//! specifically LTX-Video and Stable Video Diffusion.

pub mod go_ffi;
pub mod models;
pub mod utils;

pub use models::ltx_video::*;
