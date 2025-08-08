import Image from "next/image"
import Link from "next/link"
import { ExternalLink, Github, Linkedin, Mail, MapPin } from "lucide-react"
import FloatingCodeBackground from '../components/floating-code-background'

export default function HomePage() {
  const projects = [
    {
      title: "Predicting Churn with Machine Learning",
      description:
        "Minimizing cost of churn with 2 machine learning models (Logistic Regression and XGBoost) for a telecommunications company. The project includes data preprocessing, model training, and evaluation. Cost reduction of >75% achieved vs. baseline.",
      technologies: ["Python", "scikit-learn", "Pandas", "NumPy"],
      githubUrl: "https://github.com/FinnHet13/CodingProjects/tree/main/churn_prediction",
      liveUrl: null,
    },
    {
      title: "Sentiment Analysis with Transformers",
      description:
        "Benchmark of three transformer models (SieBERT, RoBERTa and XLNet) for sentiment analysis on a dataset of customer-agent dialogues. The project includes data cleaning, a comparison of sequential and parallel NLP processing, and model evaluation. Best accuracy of >90% achieved with SieBERT.",
      technologies: ["Python", "Transformers", "Pandas", "NumPy"],
      githubUrl: "https://github.com/FinnHet13/CodingProjects/tree/main/sentiment_analysis_bachelor_thesis",
      liveUrl: null,
    },
  ]

  return (
    <div className="min-h-screen bg-black text-white relative">
      {/* Floating Code Background */}
      <FloatingCodeBackground />
      {/* Header/Profile Section */}
      <header className="border-b border-gray-800" relative z-10>
        <div className="max-w-4xl mx-auto px-6 py-12">
          <div className="flex flex-col md:flex-row items-start gap-8">
            <div className="flex-shrink-0">
              <Image
                src="profile-pic.jpg"
                alt="Profile Picture"
                width={200}
                height={200}
                className="rounded-full border-2 border-gray-700"
              />
            </div>
            <div className="flex-1">
              <h1 className="text-4xl font-bold mb-4">Finn Hetzler</h1>
              <p className="text-xl text-gray-300 mb-6">Master in Business Analytics Student</p>
              <p className="text-gray-400 mb-6 leading-relaxed">
                Passionate about bridging business and technology. I am currently shifting my focus from BI to AI. Here, I share my (largely self-taught) coding progress. Feel free to reach out with feedback, questions, or just to connect!
              </p>
              <div className="flex flex-wrap gap-4 text-sm">
                <div className="flex items-center gap-2 text-gray-400">
                  <MapPin className="w-4 h-4" />
                  Frankfurt, Germany
                </div>
                <Link
                  href="https://github.com/FinnHet13/CodingProjects"
                  className="flex items-center gap-2 text-blue-400 hover:text-blue-300 transition-colors"
                >
                  <Github className="w-4 h-4" />
                  GitHub
                </Link>
                <Link
                  href="mailto:finn.he@protonmail.com"
                  className="flex items-center gap-2 text-blue-400 hover:text-blue-300 transition-colors"
                >
                  <Mail className="w-4 h-4" />
                  E-mail
                </Link>
                <Link
                  href="https://www.linkedin.com/in/finn-hetzler/"
                  className="flex items-center gap-2 text-blue-400 hover:text-blue-300 transition-colors"
                >
                  <Linkedin className="w-4 h-4" />
                  LinkedIn
                </Link>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Projects Section */}
      <main className="max-w-4xl mx-auto px-6 py-12 relative z-10">
        <h2 className="text-3xl font-bold mb-8">Projects</h2>
        <div className="space-y-8">
          {projects.map((project, index) => (
            <article
              key={index}
              className="border border-gray-800 rounded-lg p-6 hover:border-gray-700 transition-colors"
            >
              <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4 mb-4">
                <h3 className="text-xl font-semibold text-blue-400">{project.title}</h3>
                <div className="flex gap-3">
                  <Link
                    href={project.githubUrl}
                    className="flex items-center gap-1 text-sm text-gray-400 hover:text-white transition-colors"
                  >
                    <Github className="w-4 h-4" />
                    Code
                  </Link>
                  {project.liveUrl && (
                    <Link
                      href={project.liveUrl}
                      className="flex items-center gap-1 text-sm text-gray-400 hover:text-white transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                      Live Demo
                    </Link>
                  )}
                </div>
              </div>
              <p className="text-gray-300 mb-4 leading-relaxed">{project.description}</p>
              <div className="flex flex-wrap gap-2">
                {project.technologies.map((tech, techIndex) => (
                  <span
                    key={techIndex}
                    className="px-3 py-1 text-xs bg-gray-800 text-gray-300 rounded-full border border-gray-700"
                  >
                    {tech}
                  </span>
                ))}
              </div>
            </article>
          ))}
        </div>

        {/* Footer */}
        <footer className="mt-16 pt-8 border-t border-gray-800 text-center text-gray-500">
          <p>© 2025 Finn Hetzler. Built with Next.js and deployed on Vercel.</p>
        </footer>
      </main>
    </div>
  )
}
