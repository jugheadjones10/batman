import type { ChangeEvent } from 'react';

interface SliderProps {
  label: string;
  min: number;
  max: number;
  step?: number;
  value: number;
  onChange: (val: number) => void;
  unit?: string;
  className?: string;
  color?: string;
}

export function Slider({
  label,
  min,
  max,
  step = 1,
  value,
  onChange,
  unit = '',
  className = '',
  color = 'cyan',
}: SliderProps) {
  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    onChange(Number(e.target.value));
  };

  const colorClasses: Record<string, string> = {
    cyan: 'accent-cyan-400 text-cyan-400',
    red: 'accent-red-400 text-red-400',
    purple: 'accent-purple-400 text-purple-400',
  };

  const accentClass = colorClasses[color] || colorClasses.cyan;

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      <div className="flex justify-between items-center text-sm font-medium text-slate-300 tracking-wide">
        <span>{label}</span>
        <span className={`font-mono font-bold ${accentClass.split(' ')[1]}`}>
          {value}{unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={handleChange}
        className={`w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer ${accentClass}`}
      />
    </div>
  );
}
