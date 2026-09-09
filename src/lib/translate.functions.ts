import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";

/**
 * Batch-translate UI strings via the Google Gemini API.
 * Key stays server-side. Returns translations in the same order.
 */
export const translateTexts = createServerFn({ method: "POST" })
  .inputValidator((input: unknown) =>
    z
      .object({
        target: z.string().min(1),
        texts: z.array(z.string().min(1).max(400)).max(80),
      })
      .parse(input)
  )
  .handler(async ({ data }) => {
    if (data.target === "English") return { translations: data.texts };
    const key = process.env.GEMINI_API_KEY;
    if (!key) return { translations: data.texts };

    try {
      const res = await fetch("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${key}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "google/gemini-2.5-flash-lite",
          messages: [
            {
              role: "user",
              content:
                `Translate these mobile-app UI strings from English to ${data.target}. ` +
                `Rules: return ONLY a JSON array of strings in the exact same order and count; ` +
                `keep numbers, units (kg, km, °C, %), emojis, brand name "MOooMENTUM", and proper nouns unchanged; ` +
                `use short natural mobile-UI wording.\n\n` +
                JSON.stringify(data.texts),
            },
          ],
        }),
      });
      if (!res.ok) return { translations: data.texts };
      const json = (await res.json()) as {
        choices?: Array<{ message?: { content?: string } }>;
      };
      let raw = json.choices?.[0]?.message?.content ?? "";
      raw = raw.trim().replace(/^```(?:json)?/i, "").replace(/```$/, "").trim();
      const parsed = JSON.parse(raw) as unknown;
      if (Array.isArray(parsed) && parsed.length === data.texts.length) {
        return { translations: parsed.map((x, i) => (typeof x === "string" && x.trim() ? x : data.texts[i])) };
      }
      return { translations: data.texts };
    } catch {
      return { translations: data.texts };
    }
  });
