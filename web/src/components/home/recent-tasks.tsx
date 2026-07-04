"use client";

import * as React from "react";
import { Clock, MoreHorizontal, Trash2 } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export interface RecentProject {
  id: string;
  name: string;
  updatedAt: string;
  fileCount: number;
}

interface RecentTasksSectionProps {
  projects: RecentProject[];
  onOpen: (id: string) => void;
  onDelete?: (id: string) => void;
}

export function RecentTasksSection({
  projects,
  onOpen,
  onDelete,
}: RecentTasksSectionProps) {
  if (projects.length === 0) return null;

  return (
    <div className="mx-auto w-full max-w-2xl space-y-2">
      <div className="flex items-center gap-2 px-1">
        <Clock className="h-4 w-4 text-muted-foreground" />
        <h3 className="text-sm font-medium text-muted-foreground">
          Recent Projects
        </h3>
      </div>
      <div className="space-y-1.5">
        {projects.slice(0, 5).map((p) => (
          <Card
            key={p.id}
            className="group flex cursor-pointer items-center justify-between border-border/60 p-3 transition-colors hover:bg-accent/50"
            onClick={() => onOpen(p.id)}
          >
            <div className="flex items-center gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-muted text-xs font-bold">
                {p.name.slice(0, 2).toUpperCase()}
              </div>
              <div>
                <p className="text-sm font-medium">{p.name}</p>
                <p className="text-xs text-muted-foreground">
                  {p.fileCount} files · {p.updatedAt}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
              {onDelete && (
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(p.id);
                  }}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              )}
              <MoreHorizontal className="h-4 w-4 text-muted-foreground" />
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
