import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error(
    'Missing Supabase environment variables. Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY before loading the app.'
  )
}

// Note: Using untyped client for flexibility. Types are enforced via explicit casts.
export const supabase = createClient(
  supabaseUrl,
  supabaseAnonKey
)
