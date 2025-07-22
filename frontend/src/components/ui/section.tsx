'use client';
import { motion } from 'framer-motion';

const inView = {
    hidden: {
        opacity: 0,
        y: 200,
    },
    enter: {
        opacity: 1,
        y: 0,
        transition: {
            duration: 1,
            type: 'spring' as const,
        },
    },
};

export default function Home({ children }: { children?: React.ReactNode }) {
    return (
        <motion.div
            variants={inView}
            initial='hidden'
            whileInView='enter'
            viewport={{ once: true }}
            className='h-full'
        >
            {children}
        </motion.div>
    );
}
