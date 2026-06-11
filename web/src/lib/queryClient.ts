import { QueryClient } from '@tanstack/react-query'

// Shared singleton so non-component code (e.g. the auth store) can clear the
// cache on sign-out. Without this, another account signing in within staleTime
// would be served the previous account's cached data.
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: 1,
    },
  },
})
