/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'media',
  content: ['./src/**/*.{js,jsx,ts,tsx}'],
  presets: [require('nativewind/preset')],
  theme: {
    extend: {
      fontFamily: {
        "dm-semibold": ["DMSans_600SemiBold"]
      },
      colors: {
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: '#2A2565',
          foreground: '#FFFFFF',
        },
        secondary: {
          DEFAULT: '#FFFFFF',
          foreground: 'hsl(var(--secondary-foreground))',
        },
        destructive: 'hsl(var(--destructive))',
        accent: {
          DEFAULT: '#F59E0B',
          foreground: 'hsl(var(--accent-foreground))',
        },
        "ink": "#1F1B3A"
      },
    },
  },
  plugins: [],
};
