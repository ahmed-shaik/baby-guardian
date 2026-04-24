export default function Skeleton({ className = '', count = 1 }) {
  return (
    <>
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className={`rounded-xl bg-white/[0.04] animate-shimmer ${className}`}
        />
      ))}
    </>
  );
}
