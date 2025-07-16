// app/auto-start/page.tsx
import { redirect } from "next/navigation";

export default async function Page() {
  return redirect("/quick-verification/verify");
}
