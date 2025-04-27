import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MongoDBAtlasVectorSearch } from "@langchain/community/vectorstores/mongodb_atlas";
import { MongoClient } from "mongodb";
import * as dotenv from "dotenv";
import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";

dotenv.config();

const MONGO_URI = process.env.MONGODB_URI!;
const MONGO_DB_NAME = "rdms_rag";
const MONGO_COLLECTION_NAME = "documents";
const VECTOR_INDEX_NAME = "vector_index";

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
});

const client = new MongoClient(MONGO_URI);
await client.connect();
const db = client.db(MONGO_DB_NAME);
const collection = db.collection(MONGO_COLLECTION_NAME);

const vectorStore = new MongoDBAtlasVectorSearch(embeddings, {
  collection,
  indexName: VECTOR_INDEX_NAME,
});

const shouldUpload = process.argv.includes("--upload");

if (shouldUpload) {
  console.log("📤 Uploading pre-split documents from JSONL...");

  const chunkedDir = "./mnt/data"; // Folder containing all your .jsonl files
  const jsonlFiles = fs
    .readdirSync(chunkedDir)
    .filter((file) => file.endsWith(".jsonl"));

  let allSplits: Document[] = [];

  for (const file of jsonlFiles) {
    console.log(`🔍 Loading ${file}`);
    const filePath = path.join(chunkedDir, file);

    const rawLines = fs
      .readFileSync(filePath, "utf8")
      .split("\n")
      .filter(Boolean);

    const docs = rawLines.map((line) => {
      const { text, metadata } = JSON.parse(line);
      return new Document({
        pageContent: text,
        metadata: metadata,
      });
    });

    allSplits.push(...docs);
  }

  console.log(`✅ Loaded ${allSplits.length} documents from JSONL files`);

  await vectorStore.addDocuments(allSplits);
  console.log(`✅ Uploaded ${allSplits.length} documents to the vector store.`);
}

const promptTemplate =
  ChatPromptTemplate.fromTemplate(`You are a helpful assistant. Use the context below to answer the question.
If the answer cannot be found in the context, respond with "I don't know."

Context:
{context}

Question: {question}
Answer:`);

const InputStateAnnotation = Annotation.Root({
  question: Annotation<string>,
});

const StateAnnotation = Annotation.Root({
  question: Annotation<string>,
  context: Annotation<Document[]>,
  answer: Annotation<string>,
});

const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
});

const retrieve = async (state: typeof InputStateAnnotation.State) => {
  const retrievedDocsWithScores = await vectorStore.similaritySearchWithScore(
    state.question,
    4
  );

  console.log("🔍 Retrieved docs with scores:");
  retrievedDocsWithScores.forEach(([doc, score], i) => {
    console.log(
      `Doc ${i + 1} (score: ${score.toFixed(2)}): ${doc.pageContent.slice(
        0,
        200
      )}`
    );
  });

  const retrievedDocs = retrievedDocsWithScores.map(([doc]) => doc);
  return { context: retrievedDocs };
};

const generate = async (state: typeof StateAnnotation.State) => {
  console.log("🧪 context doc count:", state.context?.length);

  const docsContent = (state.context || [])
    .map((doc) => doc.pageContent || "")
    .filter(Boolean)
    .join("\n");

  if (!docsContent.trim()) {
    return { answer: "I couldn't find anything relevant in the documents." };
  }

  console.log(
    "\n🧠 Final context for the prompt:\n",
    docsContent.slice(0, 1000)
  );

  const messages = await promptTemplate.invoke({
    question: state.question,
    context: docsContent,
  });

  const response = await llm.invoke(messages);
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

console.log("🧠 Ready!");
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
