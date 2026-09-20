import { useEffect, useRef } from "react";
import { motion, useMotionValue, useTransform, animate } from "framer-motion";

// Smoothly tweens a displayed number whenever `value` changes — used for
// vitals and the SOFA score so every new 30s reading feels alive rather
// than snapping.
export default function AnimatedNumber({ value, decimals = 0, className = "", duration = 0.8 }) {
  const motionVal = useMotionValue(value);
  const rounded = useTransform(motionVal, (v) => v.toFixed(decimals));
  const prev = useRef(value);

  useEffect(() => {
    const controls = animate(motionVal, value, {
      duration,
      ease: [0.16, 1, 0.3, 1],
    });
    prev.current = value;
    return controls.stop;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return <motion.span className={className}>{rounded}</motion.span>;
}
