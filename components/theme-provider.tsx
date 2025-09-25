'use client'

import * as React from 'react'
import {
  ThemeProvider as NextThemesProvider,
  type ThemeProviderProps,
} from 'next-themes'

interface ExtendedThemeProviderProps extends Omit<ThemeProviderProps, 'themes'> {
  children: React.ReactNode
}

export function ThemeProvider({ children, ...props }: ExtendedThemeProviderProps) {
  return (
    <NextThemesProvider 
      {...props}
      themes={['light', 'dark', 'black']}
      defaultTheme="light"
      enableSystem={false}
      storageKey="socialguard-theme"
      attribute="class"
      disableTransitionOnChange={false}
    >
      {children}
    </NextThemesProvider>
  )
}
