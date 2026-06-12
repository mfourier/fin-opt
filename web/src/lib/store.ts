import { create } from 'zustand'
import { User } from '@supabase/supabase-js'
import { supabase } from './supabase'
import { queryClient } from './queryClient'

interface AuthState {
  user: User | null
  loading: boolean
  signIn: (email: string, password: string) => Promise<void>
  signUp: (email: string, password: string) => Promise<void>
  signInWithGoogle: () => Promise<void>
  signOut: () => Promise<void>
  initialize: () => Promise<void>
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  loading: true,

  initialize: async () => {
    try {
      const { data: { session } } = await supabase.auth.getSession()
      set({ user: session?.user ?? null, loading: false })

      // Listen for auth changes
      supabase.auth.onAuthStateChange((event, session) => {
        set({ user: session?.user ?? null })
        // Drop all cached data on sign-out so the next account never sees the
        // previous account's profiles/scenarios/jobs while they are still
        // "fresh" within staleTime. Covers explicit sign-out and expiry.
        if (event === 'SIGNED_OUT') {
          queryClient.clear()
        }
      })
    } catch (error) {
      console.error('Error initializing auth:', error)
      set({ loading: false })
    }
  },

  signIn: async (email: string, password: string) => {
    const { error } = await supabase.auth.signInWithPassword({
      email,
      password,
    })
    if (error) throw error
  },

  signUp: async (email: string, password: string) => {
    const { error } = await supabase.auth.signUp({
      email,
      password,
    })
    if (error) throw error
  },

  signInWithGoogle: async () => {
    // Real Supabase OAuth. Requires the Google provider to be enabled in the
    // Supabase project; if it is not, Supabase returns a descriptive error
    // which the caller surfaces (no fake/placeholder logic here).
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: { redirectTo: window.location.origin },
    })
    if (error) throw error
  },

  signOut: async () => {
    const { error } = await supabase.auth.signOut()
    if (error) throw error
    set({ user: null })
  },
}))

// Initialize auth on app load
useAuthStore.getState().initialize()
