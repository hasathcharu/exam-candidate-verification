import {motion, AnimatePresence} from 'framer-motion';
export default function Orb() {
  return (
    <AnimatePresence>
        <motion.div className='flex items-center justify-center mb-1'>
            <motion.div className='wave-wrapper'>
              <motion.div className='wave one'></motion.div>
              <motion.div className='wave two'></motion.div>
              <motion.div className='wave three'></motion.div>
              <motion.div className='wave four'></motion.div>
            </motion.div>
        </motion.div>
    </AnimatePresence>
  );
}
