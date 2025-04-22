import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { MongoDBAtlasVectorSearch } from "@langchain/community/vectorstores/mongodb_atlas";
import readline from "readline";

import * as dotenv from "dotenv";
import * as fs from "fs";
import * as path from "path";
import { MongoClient } from "mongodb";

dotenv.config();

const MONGO_URI = process.env.MONGODB_URI || "";
const MONGO_DB_NAME = "rmd_rag";
const MONGO_COLLECTION_NAME = "documents";

const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
});

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
});

const DOCS_DIR = "./docs";

const client = new MongoClient(MONGO_URI);
await client.connect();
const collection = client.db(MONGO_DB_NAME).collection(MONGO_COLLECTION_NAME);

const shouldUpload = process.argv.includes("--upload");
let vectorStore;

if (shouldUpload) {
  console.log("📤 Uploading and indexing documents...");
  const files = fs
    .readdirSync(DOCS_DIR)
    .filter((file) => fs.statSync(path.join(DOCS_DIR, file)).isFile());

  let allSplits: Document[] = [];
  for (const file of files) {
    const filePath = path.join(DOCS_DIR, file);
    console.log(`🔍 Processing: ${filePath}`);

    const loader = new PDFLoader(filePath);
    const docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 100,
    });

    const splits = await splitter.splitDocuments(docs);
    allSplits.push(...splits);
  }

  // Create or update vector store
  vectorStore = await MongoDBAtlasVectorSearch.fromDocuments(
    allSplits,
    embeddings,
    {
      collection, // 👈 must pass actual collection
      indexName: "vector_index",
    }
  );
} else {
  vectorStore = new MongoDBAtlasVectorSearch(embeddings, {
    collection,
    indexName: "vector_index",
  });
}

// Setup prompt and RAG state
const promptTemplate = await pull<ChatPromptTemplate>("rlm/rag-prompt");

const InputStateAnnotation = Annotation.Root({
  question: Annotation<string>,
});

const StateAnnotation = Annotation.Root({
  question: Annotation<string>,
  context: Annotation<Document[]>,
  answer: Annotation<string>,
});

const retrieve = async (state: typeof InputStateAnnotation.State) => {
  const retrievedDocsWithScores = await vectorStore.similaritySearchWithScore(
    state.question,
    4
  );

  const retrievedDocs = retrievedDocsWithScores.map(([doc]) => doc);
  return { context: retrievedDocs };
};

const generate = async (state: typeof StateAnnotation.State) => {
  const docsContent = (state.context || [])
    .map((doc) => doc.pageContent || "")
    .filter(Boolean)
    .join("\n");

  const prompt = `You are a helpful assistant. Answer the question using ONLY the provided context below.
  If the answer cannot be found in the context, respond with "I don't know."
  Context:
  ${docsContent}

  Question: ${state.question}
  Answer:`;

  const messages = await promptTemplate.invoke({
    prompt,
    question: state.question,
    context: docsContent,
  });
  const response = await llm.invoke(messages);
  console.log("generate() context size:", state.context?.length);

  return { answer: response.content };
};

const graph = new StateGraph(StateAnnotation)
  .addNode("retrieve", retrieve)
  .addNode("generate", generate)
  .addEdge("__start__", "retrieve")
  .addEdge("retrieve", "generate")
  .addEdge("generate", "__end__")
  .compile();

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  prompt: "💬 Ask a question (or type 'exit'): ",
});

console.log("🧠 Ready! Documents indexed.");

// --------------------------------------------------------

rl.prompt();

rl.on("line", async (line) => {
  const question = line.trim();
  if (question.toLowerCase() === "exit") {
    rl.close();
    return;
  }

  try {
    const result = await graph.invoke({ question });
    console.log(`🗣️  ${result.answer}`);
  } catch (error: any) {
    console.error("❌ Failed to answer question:", error?.message || error);
  }

  rl.prompt();
});

// MongoDB search vector index
// {
//   "fields": [
//     {
//       "type": "vector",
//       "path": "embedding",
//       "numDimensions": 1536,
//       "similarity": "cosine"
//     }
//   ]
// }
