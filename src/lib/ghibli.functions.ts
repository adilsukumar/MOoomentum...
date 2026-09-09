import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";

/**
 * Convert an uploaded dog photo into a Ghibli-style portrait
 * using the Google Gemini API (Gemini image model). Key stays server-side.
 */
export const convertToGhibli = createServerFn({ method: "POST" })
  .inputValidator((input: unknown) =>
    z
      .object({
        base64: z.string().min(1),
        mime: z.string().default("image/jpeg"),
      })
      .parse(input)
  )
  .handler(async ({ data }) => {
    const key = process.env.GEMINI_API_KEY;
    if (!key) throw new Error("Missing GEMINI_API_KEY");

    const dataUrl = `data:${data.mime};base64,${data.base64}`;

    const res = await fetch("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${key}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash-image",
        modalities: ["image", "text"],
        messages: [
          {
            role: "user",
            content: [
              {
                type: "text",
                text:
                  "Transform this dog photo into a Studio Ghibli style portrait. " +
                  "Soft hand-painted watercolor textures, warm gentle lighting, " +
                  "kawaii proportions, soft pastel palette, dreamy painterly background. " +
                  "Keep the dog clearly recognisable (same breed, fur color, pose). " +
                  "Return only the image.",
              },
              { type: "image_url", image_url: { url: dataUrl } },
            ],
          },
        ],
      }),
    });

    if (!res.ok) {
      const text = await res.text();
      if (res.status === 429) throw new Error("Rate limited. Please try again in a moment.");
      if (res.status === 402) throw new Error("AI credits exhausted. Add credits in Workspace settings.");
      throw new Error(`Ghibli generation failed: ${res.status} ${text}`);
    }

    const json = (await res.json()) as {
      choices?: Array<{ message?: { images?: Array<{ image_url?: { url?: string } }> } }>;
    };

    const url = json.choices?.[0]?.message?.images?.[0]?.image_url?.url;
    if (!url) throw new Error("No image returned from AI gateway");

    return { url };
  });
