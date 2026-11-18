import { clsx, type ClassValue } from 'clsx';
import Constants from 'expo-constants';
import { twMerge } from 'tailwind-merge';

const API_HOST = Constants.expoConfig?.hostUri?.split(':')[0] ?? '127.0.0.1';
const API_URL = process.env.EXPO_PUBLIC_API_URL ?? `http://${API_HOST}:8765/`;

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export async function get<T = unknown>(endpoint: string, params: URLSearchParams): Promise<T> {
  const response = await fetch(`${API_URL}${endpoint}?${params.toString()}`);
  if (!response.ok) {
    throw new Error(`GET ${endpoint} failed with status ${response.status}`);
  }
  return response.json();
}

type PostOptions = {
  invalidMessage?: string;
  token?: string | null;
};

export async function post<T = unknown>(
  endpoint: string,
  body: unknown,
  {
    invalidMessage = 'Username must be 3-32 letters, numbers or underscores, and the password at least 8 characters',
    token,
  }: PostOptions = {}
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error('Could not reach the server');
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data?.detail;
    throw new Error(typeof detail === 'string' ? detail : invalidMessage);
  }
  return data as T;
}
