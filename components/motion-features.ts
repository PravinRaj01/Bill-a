import { domMax } from "motion/react";

// The full feature set: domAnimation plus drag (the intro's swipe) and layout animations (the sliding pill
// in the bottom nav and the segmented controls). Loaded on demand by MotionProvider, so it isn't in the
// initial bundle.
export default domMax;
