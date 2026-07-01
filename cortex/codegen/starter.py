"""
cortex/codegen/starter.py — Project scaffolding templates.

Provides starter file sets for different project types:
  - next: Full Next.js 14 App Router with Tailwind, shadcn tokens, lucide, framer-motion
  - minimal: Minimal Next.js 14 with Tailwind
  - express: Express.js with TypeScript

Each starter is a dict of path → file content, ready to write to disk.
"""

from __future__ import annotations

from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Next.js starter (full — Tailwind, tokens, lucide, framer-motion)
# ---------------------------------------------------------------------------

NEXT_STARTER: Dict[str, str] = {
    "package.json": """\
{
  "name": "app",
  "version": "0.0.1",
  "private": true,
  "scripts": {
    "dev": "next dev --hostname 0.0.0.0 --port 3000",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "clsx": "2.1.1",
    "framer-motion": "11.11.17",
    "lucide-react": "0.453.0",
    "next": "14.2.18",
    "react": "18.3.1",
    "react-dom": "18.3.1",
    "tailwind-merge": "2.5.4"
  },
  "devDependencies": {
    "@types/node": "20.14.10",
    "@types/react": "18.3.3",
    "@types/react-dom": "18.3.0",
    "autoprefixer": "10.4.20",
    "postcss": "8.4.47",
    "tailwindcss": "3.4.14",
    "typescript": "5.5.4"
  }
}
""",
    "tsconfig.json": """\
{
  "compilerOptions": {
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "baseUrl": ".",
    "paths": { "@/*": ["./*"] },
    "plugins": [{ "name": "next" }]
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
""",
    "next-env.d.ts": """\
/// <reference types="next" />
/// <reference types="next/image-types/global" />
""",
    "next.config.mjs": """\
/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      { protocol: "https", hostname: "images.unsplash.com" },
      { protocol: "https", hostname: "placehold.co" },
    ],
  },
};
export default nextConfig;
""",
    "postcss.config.mjs": """\
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
""",
    "tailwind.config.ts": """\
import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{ts,tsx,mdx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    container: {
      center: true,
      padding: "1.5rem",
      screens: { "2xl": "1400px" },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
export default config;
""",
    "app/globals.css": """\
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: 0 0% 100%;
    --foreground: 240 10% 3.9%;
    --card: 0 0% 100%;
    --card-foreground: 240 10% 3.9%;
    --popover: 0 0% 100%;
    --popover-foreground: 240 10% 3.9%;
    --primary: 240 5.9% 10%;
    --primary-foreground: 0 0% 98%;
    --secondary: 240 4.8% 95.9%;
    --secondary-foreground: 240 5.9% 10%;
    --muted: 240 4.8% 95.9%;
    --muted-foreground: 240 3.8% 46.1%;
    --accent: 240 4.8% 95.9%;
    --accent-foreground: 240 5.9% 10%;
    --destructive: 0 72.2% 50.6%;
    --destructive-foreground: 0 0% 98%;
    --border: 240 5.9% 90%;
    --input: 240 5.9% 90%;
    --ring: 240 5.9% 10%;
    --radius: 0.75rem;
  }

  .dark {
    --background: 240 10% 3.9%;
    --foreground: 0 0% 98%;
    --card: 240 10% 3.9%;
    --card-foreground: 0 0% 98%;
    --popover: 240 10% 3.9%;
    --popover-foreground: 0 0% 98%;
    --primary: 0 0% 98%;
    --primary-foreground: 240 5.9% 10%;
    --secondary: 240 3.7% 15.9%;
    --secondary-foreground: 0 0% 98%;
    --muted: 240 3.7% 15.9%;
    --muted-foreground: 240 5% 64.9%;
    --accent: 240 3.7% 15.9%;
    --accent-foreground: 0 0% 98%;
    --destructive: 0 62.8% 30.6%;
    --destructive-foreground: 0 0% 98%;
    --border: 240 3.7% 15.9%;
    --input: 240 3.7% 15.9%;
    --ring: 240 4.9% 83.9%;
  }
}

@layer base {
  * { @apply border-border; }
  body {
    @apply bg-background text-foreground antialiased;
    font-feature-settings: "rlig" 1, "calt" 1;
  }
}
""",
    "lib/utils.ts": """\
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
""",
    "app/layout.tsx": """\
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Your app",
  description: "Built with Odysseus.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="font-sans">{children}</body>
    </html>
  );
}
""",
    "app/page.tsx": """\
export default function Page() {
  return (
    <main className="relative min-h-screen overflow-hidden bg-gradient-to-b from-background to-muted">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 -top-40 h-[28rem] bg-gradient-to-b from-primary/10 to-transparent blur-3xl"
      />
      <section className="relative mx-auto flex min-h-screen max-w-3xl flex-col items-center justify-center px-6 py-24 text-center">
        <span className="mb-6 inline-flex items-center rounded-full border border-border bg-background/60 px-3 py-1 text-xs font-medium text-muted-foreground backdrop-blur">
          Odysseus Codegen
        </span>
        <h1 className="text-5xl font-semibold tracking-tight md:text-6xl">
          Your app starts here
        </h1>
        <p className="mt-6 max-w-xl text-lg leading-relaxed text-muted-foreground">
          Describe what you want to build. Your project takes shape file by file.
        </p>
      </section>
    </main>
  );
}
""",
}

# ---------------------------------------------------------------------------
# Minimal Next.js starter (no luxe deps)
# ---------------------------------------------------------------------------

NEXT_MINIMAL_STARTER: Dict[str, str] = {
    "package.json": """\
{
  "name": "app-minimal",
  "version": "0.0.1",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start"
  },
  "dependencies": {
    "next": "14.2.18",
    "react": "18.3.1",
    "react-dom": "18.3.1"
  },
  "devDependencies": {
    "@types/node": "20.14.10",
    "@types/react": "18.3.3",
    "@types/react-dom": "18.3.0",
    "autoprefixer": "10.4.20",
    "postcss": "8.4.47",
    "tailwindcss": "3.4.14",
    "typescript": "5.5.4"
  }
}
""",
    "tsconfig.json": """\
{
  "compilerOptions": {
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "baseUrl": ".",
    "paths": { "@/*": ["./*"] }
  },
  "include": ["**/*.ts", "**/*.tsx"],
  "exclude": ["node_modules"]
}
""",
    "next.config.mjs": "const nextConfig = {};\nexport default nextConfig;\n",
    "postcss.config.mjs": """\
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
""",
    "tailwind.config.ts": """\
import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx,mdx}"],
  theme: {
    extend: {},
  },
  plugins: [],
};
export default config;
""",
    "app/globals.css": """\
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  font-family: system-ui, sans-serif;
}
""",
    "app/layout.tsx": """\
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "My App",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
""",
    "app/page.tsx": """\
export default function Page() {
  return (
    <main>
      <h1>Hello, world!</h1>
    </main>
  );
}
""",
}

# ---------------------------------------------------------------------------
# Express.js + TypeScript starter
# ---------------------------------------------------------------------------

EXPRESS_STARTER: Dict[str, str] = {
    "package.json": """\
{
  "name": "api",
  "version": "0.0.1",
  "private": true,
  "scripts": {
    "dev": "tsx watch src/index.ts",
    "build": "tsc",
    "start": "node dist/index.js"
  },
  "dependencies": {
    "cors": "^2.8.5",
    "express": "^4.21.0",
    "helmet": "^7.1.0"
  },
  "devDependencies": {
    "@types/cors": "^2.8.17",
    "@types/express": "^4.17.21",
    "@types/node": "^20.14.0",
    "tsx": "^4.19.0",
    "typescript": "^5.5.0"
  }
}
""",
    "tsconfig.json": """\
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "commonjs",
    "lib": ["ES2022"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true
  },
  "include": ["src"],
  "exclude": ["node_modules", "dist"]
}
""",
    ".gitignore": """\
node_modules/
dist/
.env
""",
    "src/index.ts": """\
import express from "express";
import cors from "cors";
import helmet from "helmet";

const app = express();
const port = process.env.PORT || 3001;

app.use(helmet());
app.use(cors());
app.use(express.json());

app.get("/health", (_req, res) => {
  res.json({ status: "ok" });
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
""",
}

# ---------------------------------------------------------------------------
# Starter registry
# ---------------------------------------------------------------------------

_STARTERS: Dict[str, Dict[str, str]] = {
    "next": NEXT_STARTER,
    "next-minimal": NEXT_MINIMAL_STARTER,
    "express": EXPRESS_STARTER,
}


def get_starter(name: str) -> Optional[Dict[str, str]]:
    """Get a starter template by name.

    Returns None if the starter is not found.
    """
    return _STARTERS.get(name)


def list_starters() -> list[str]:
    """List available starter template names."""
    return list(_STARTERS.keys())


def is_placeholder_file(starter_name: str, path: str, content: str) -> bool:
    """Check if a file path+content matches a starter's placeholder.

    Used by the context loader to flag files that may still hold
    unmodified scaffold content.
    """
    starter = _STARTERS.get(starter_name)
    if starter is None:
        return False
    expected = starter.get(path)
    if expected is None:
        return False
    return content == expected
