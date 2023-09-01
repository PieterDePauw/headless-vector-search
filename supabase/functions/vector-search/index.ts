import "xhr";
import "xhr_polyfill";
import GPT3Tokenizer from "gpt3-tokenizer";
import { createClient } from "@supabase/supabase-js";
import { codeBlock, oneLine } from "commmon-tags";
import { Configuration, OpenAIApi } from "openai";
import { ensureGetEnv } from "../_utils/env.ts";
import { ApplicationError, UserError } from "../_utils/errors.ts";

// Define CORS headers
const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
}

const corsStreamingHeaders = {
  ...corsHeaders,
  "Content-Type": "text/event-stream",
}

// Load environment variables
const NEXT_PUBLIC_SUPABASE_URL = ensureGetEnv("NEXT_PUBLIC_SUPABASE_URL") as string || "https://qzmvxngzbugwtgvyfgpj.supabase.co"
// const NEXT_PUBLIC_SUPABASE_ANON_KEY = ensureGetEnv("NEXT_PUBLIC_SUPABASE_ANON_KEY") as string
const SERVICE_ROLE_KEY = ensureGetEnv("SERVICE_ROLE_KEY") as string
const OPENAI_API_KEY = ensureGetEnv("OPENAI_API_KEY") as string
const OPENAI_EMBEDDINGS_MODEL = ensureGetEnv("OPENAI_EMBEDDINGS_MODEL") as string || "text-embedding-ada-002"
const OPENAI_COMPLETIONS_MODEL = ensureGetEnv("OPENAI_COMPLETIONS_MODEL") as string || "gpt-3.5-turbo"
const OPENAI_COMPLETIONS_ENDPOINT = ensureGetEnv("OPENAI_COMPLETIONS_ENDPOINT") as string || "https://api.openai.com/v1/chat/completions"

// Initialize Supabase client
const supabaseClient = createClient(NEXT_PUBLIC_SUPABASE_URL, SERVICE_ROLE_KEY, {
  db: { schema: 'docs' }
})
const openAiConfiguration = new Configuration({ apiKey: OPENAI_API_KEY })
const openai = new OpenAIApi(openAiConfiguration)
// const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

Deno.serve(async (req) => {
  try {
    // Get the request method and return an empty response for OPTIONS requests
    if (req.method === "OPTIONS") {
      return new Response("ok", { headers: corsHeaders })
    }

    // Get the query from the request data
    const query = new URL(req.url).searchParams.get("query")
    if (!query) {
      throw new UserError("Missing query in request data")
    }

    // Moderate the content to comply with OpenAI API Terms of Service
    const sanitizedQuery = query.trim().replaceAll("\n", " ")
    const moderationResponse = await openai.createModeration({ input: sanitizedQuery })
    const [results] = moderationResponse.data.results
    if (results.flagged) {
      throw new UserError("Flagged content", { flagged: true, categories: results.categories })
    }

    // Create an embedding for the input
    const embeddingResponse = await openai.createEmbedding({
      model: OPENAI_EMBEDDINGS_MODEL,
      input: sanitizedQuery,
    });

    // Throw an error if the embedding request failed
    if (embeddingResponse.status !== 200) {
      throw new ApplicationError("Failed to create embedding for question", embeddingResponse);
    }

    // Match the embedding to the most similar page sections
    const [{ embedding }] = embeddingResponse.data.data;
    const { error: matchError, data: pageSections } = await supabaseClient.rpc(
      "match_page_sections",
      {
        embedding,
        match_threshold: 0.78,
        match_count: 10,
        min_content_length: 50,
      }
    );

    // Throw an error if the page section matching failed
    if (matchError) {
      throw new ApplicationError("Failed to match page sections", matchError);
    }

    // Initialize the tokenizer and define variables for the token count and the context text
    const tokenizer = new GPT3Tokenizer({ type: "gpt3" });
    let tokenCount = 0;
    let contextText = "";

    // Loop over the page sections and add them to the context text until the maximum token count is reached
    for (const pageSection of pageSections) {
      const currentDocument = pageSection.content as string || "";
      const { text: currentDocumentInTokens } = tokenizer.encode(currentDocument);
      const currentDocumentTokenCount = currentDocumentInTokens.length;
      const currentDocumentText = `${currentDocument.trim()}\n---\n`

      tokenCount += currentDocumentTokenCount;
      // deno-lint-ignore no-extra-semi
      if (tokenCount >= 1500) { break };
      contextText += currentDocumentText;
    }

    // Generate a prompt for the OpenAI API
    const prompt = codeBlock`
      ${oneLine`
        You are a very enthusiastic support representative who loves
        to help people! Given the following sections from the product
        documentation, answer the question using only that information,
        outputted in markdown format. If you are unsure and the answer
        is not explicitly written in the documentation, say
        "Sorry, I don't know how to help with that."
      `}

      Context sections:
      ${contextText}

      Question: """
      ${sanitizedQuery}
      """

      Answer as markdown (including related code snippets if available):
    `;

    // Create the completion request for the OpenAI API
    const completionOptions = {
      messages: [
        { role: "system", content: prompt },
        { role: "user", content: query }
      ],
      model: OPENAI_COMPLETIONS_MODEL,
      max_tokens: 512,
      temperature: 0,
      stream: true,
    };

    // The Fetch API allows for easier response streaming over the OpenAI client.
    const response = await fetch(OPENAI_COMPLETIONS_ENDPOINT, {
      headers: {
        "Authorization": `Bearer ${OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      method: "POST",
      body: JSON.stringify(completionOptions),
    });

    // Throw an error if the completion request failed
    if (!response.ok) {
      const error = await response.json();
      throw new ApplicationError("Failed to generate completion", error);
    }

    // Stream the response from the OpenAI API to the client
    return new Response(response.body, { headers: corsStreamingHeaders });

  } catch (err: unknown) {
    if (err instanceof UserError) {
      return Response.json(
        {
          error: err.message,
          data: err.data,
        },
        {
          status: 400,
          headers: corsHeaders,
        }
      );
    } else if (err instanceof ApplicationError) {
      // Print out application errors with their additional data
      console.error(`${err.message}: ${JSON.stringify(err.data)}`);
    } else {
      // Print out unexpected errors as is to help with debugging
      console.error(err);
    }

    // TODO: include more response info in debug environments
    return Response.json(
      {
        error: "There was an error processing your request",
      },
      {
        status: 500,
        headers: corsHeaders,
      }
    );
  }
});