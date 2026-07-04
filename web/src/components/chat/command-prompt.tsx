"use client";

import * as React from "react";
import { ArrowRight, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

interface CommandPromptProps {
  onSubmit: (prompt: string) => void;
  loading?: boolean;
  placeholder?: string;
}

export function CommandPrompt({
  onSubmit,
  loading,
  placeholder = "Describe the app you want to build...",
}: CommandPromptProps) {
  const [value, setV] = React.useState("");

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      if (value.trim() && !loading) {
        onSubmit(value.trim());
        setV("");
      }
    }
  };

  return (
    <div className="relative mx-auto w-full max-w-2xl">
      <div className="absolute -inset-2 rounded-3xl bg-gradient-to-br from-primary/10 via-transparent to-transparent blur-2xl" />
      <div className="relative flex flex-col gap-3 rounded-2xl border border-border bg-card p-4 shadow-lg">
        <Textarea
          value={value}
          onChange={(e) => setV(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="min-h-[100px] resize-none border-0 bg-transparent p-0 shadow-none focus-visible:ring-0"
          disabled={loading}
        />
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">
            {value.trim() ? "⌘ + Enter to build" : "Type your idea above"}
          </span>
          <Button
            onClick={() => {
              if (value.trim() && !loading) {
                onSubmit(value.trim());
                setV("");
              }
            }}
            disabled={!value.trim() || loading}
            size="sm"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Building...
              </>
            ) : (
              <>
                Build
                <ArrowRight className="h-4 w-4" />
              </>
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}
