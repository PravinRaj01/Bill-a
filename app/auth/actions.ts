'use server'

import { revalidatePath } from 'next/cache'
import { redirect } from 'next/navigation'
import { createClient } from '@/utils/supabase/server'

export async function login(formData: FormData) {
  const supabase = await createClient()

  // 1. Get data from form
  const email = formData.get('email') as string
  const password = formData.get('password') as string

  // 2. Sign in
  const { error } = await supabase.auth.signInWithPassword({
    email,
    password,
  })

  if (error) {
    return { error: error.message }
  }

  // 3. Redirect to dashboard
  revalidatePath('/', 'layout')
  redirect('/dashboard')
}

export async function signup(formData: FormData) {
  const supabase = await createClient()

  const email = formData.get('email') as string
  const password = formData.get('password') as string
  
  // 1. Sign up
  const { error } = await supabase.auth.signUp({
    email,
    password,
  })

  if (error) {
    return { error: error.message }
  }

  // 2. Redirect
  revalidatePath('/', 'layout')
  redirect('/dashboard')
}

export async function saveGroup(name: string, namesList: string[]) {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()

  if (!user) return { error: "Login required to save groups" }

  const { error } = await supabase
    .from('saved_groups')
    .insert({
      user_id: user.id,
      group_name: name,
      names: namesList
    })

  if (error) return { error: error.message }
  revalidatePath('/dashboard')
}